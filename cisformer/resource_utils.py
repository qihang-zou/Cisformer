from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

SUPPORTED_SPECIES = ("human", "mouse")

ANNOTATION_URLS = {
    "human": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.primary_assembly.annotation.gtf.gz",
    "mouse": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M39/gencode.vM39.primary_assembly.annotation.gtf.gz",
}

ANNOTATION_FILENAMES = {
    "human": "gencode.v49.primary_assembly.annotation.gtf.gz",
    "mouse": "gencode.vM39.primary_assembly.annotation.gtf.gz",
}


def validate_species(species):
    if species not in SUPPORTED_SPECIES:
        raise ValueError("Species should be human or mouse.")


def local_resource_dir():
    path = Path("cisformer_config") / "resource"
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_annotation_path(species):
    validate_species(species)
    return local_resource_dir() / ANNOTATION_FILENAMES[species]


def download_annotation(species):
    validate_species(species)
    output = local_annotation_path(species)
    if output.exists():
        print(f"Previous {species} genome annotation found: {output}")
        return output

    url = ANNOTATION_URLS[species]
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    print(f"Downloading {species} genome annotation from {url}")
    print(f"Saving to {output}")
    try:
        urlretrieve(url, tmp_output)
        tmp_output.replace(output)
    except (OSError, URLError) as exc:
        if tmp_output.exists():
            tmp_output.unlink()
        raise RuntimeError(
            f"Failed to download {species} genome annotation. "
            f"Please download it manually from {url} and place it at {output}."
        ) from exc
    return output


def require_annotation(species):
    path = local_annotation_path(species)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {species} genome annotation: {path}. "
            "Run `cisformer generate_default_config --species "
            f"{species}` to download it, or download it manually from "
            f"{ANNOTATION_URLS[species]} and place it at {path}."
        )
    return path


def gene_surround_path(species, extend_kbp, idx=False):
    validate_species(species)
    suffix = "_idx" if idx else ""
    return local_resource_dir() / f"{species}_gene_surround_enhancers_{extend_kbp}kbp{suffix}.pkl"
