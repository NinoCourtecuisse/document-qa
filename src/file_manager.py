from fastapi import UploadFile, File, HTTPException
from pathlib import Path
import hashlib
import shutil
import pickle

class FileManager:
    def __init__(self, persistent_path: Path,
                 file_names: Path, file_hashes: Path):
        self.persistent_storage = persistent_path
        self.file_names = file_names
        self.file_hashes = file_hashes

        self.buffer_storage = Path("./buffer")

    def add_to_buffer(self, file: UploadFile = File(...)) -> None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        shutil.rmtree(self.buffer_storage)
        self.buffer_storage.mkdir(exist_ok=True)
        with open(self.buffer_storage / file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    def validate_file(self, file_name: str | None) -> None:
        if not file_name:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_extension = Path(file_name).suffix.lower()
        if file_extension != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        try:
            with open(self.file_names, "rb") as f:
                file_names = pickle.load(f)
        except (EOFError, FileNotFoundError):
            file_names = set()
        if file_name in file_names:
            raise HTTPException(status_code=400, detail="File with same name already exists")

        try:
            with open(self.file_hashes, "rb") as f:
                file_hashes = pickle.load(f)
        except (EOFError, FileNotFoundError):
            file_hashes = set()
        file_hash = self.compute_hash(self.buffer_storage / file_name)
        if file_hash in file_hashes:
            raise HTTPException(status_code=400, detail="Document with same content already exists")

    def store(self, file: UploadFile = File(...)) -> None:
        if file.filename:
            # Copy from buffer instead of upload stream (which is already consumed)
            buffer_path = self.buffer_storage / file.filename
            with open(buffer_path, "rb") as buffer_file:
                with open(self.persistent_storage / file.filename, "wb") as persistent:
                    shutil.copyfileobj(buffer_file, persistent)

            try:
                with open(self.file_names, "rb") as f:
                    file_names = pickle.load(f)
            except (EOFError, FileNotFoundError):
                file_names = set()
            file_names.add(file.filename)
            with open(self.file_names, "wb") as f:
                pickle.dump(file_names, f)

            try:
                with open(self.file_hashes, "rb") as f:
                    file_hashes = pickle.load(f)
            except (EOFError, FileNotFoundError):
                file_hashes = set()
            file_hashes.add(self.compute_hash(self.persistent_storage / file.filename))
            with open(self.file_hashes, "wb") as f:
                pickle.dump(file_hashes, f)

    def list_documents(self) -> dict:
        documents = []
        for file_path in self.persistent_storage.glob("*"):
            documents.append({
                "path": str(file_path),
            })
        return {"documents": documents}
    
    def clear_files(self) -> int:
        files_deleted = 0
        for file_path in self.persistent_storage.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                files_deleted += 1
        self.file_hashes.unlink(missing_ok=True)
        self.file_names.unlink(missing_ok=True)
        self.file_hashes.touch()
        self.file_names.touch()
        return files_deleted

    def compute_hash(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, mode="rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
