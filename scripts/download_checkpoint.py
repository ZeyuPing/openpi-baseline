#!/usr/bin/env python3
import os
import sys
import pathlib
import subprocess
import concurrent.futures
import urllib.request
import urllib.error

# The public GCS bucket base url for openpi-assets
GCS_BASE_URL = "https://storage.googleapis.com/openpi-assets"
DEFAULT_CHECKPOINT = "checkpoints/pi05_base/params"

# Hardcoded list of files for checkpoints/pi05_base/params to avoid GCS API listing issues
PI05_BASE_FILES = [
    "_CHECKPOINT_METADATA",
    "_METADATA",
    "_sharding",
    "array_metadatas/process_0",
    "commit_success.txt",
    "d/1c4302d2d2000b5f3eb4fa1350fdef9a",
    "manifest.ocdbt",
    "ocdbt.process_0/d/0832cad6c37f82d4eedd897dcbb8da9d",
    "ocdbt.process_0/d/247d4b7c814d8b1a23fa8a20f36a88f7",
    "ocdbt.process_0/d/35a545f74995e511808d4f94dfbef3b6",
    "ocdbt.process_0/d/73bbae8a4deba6498bf07d96b215a574",
    "ocdbt.process_0/d/7bc9d3296d23a6fb83a6b3778ac6e964",
    "ocdbt.process_0/d/7d78f38c5d8be1eea31406644dde9bd6",
    "ocdbt.process_0/d/828bee85475e37c61e1cc19e32d1c5ef",
    "ocdbt.process_0/d/8c5d7070ea57bdce2f0a19f95b8a21b4",
    "ocdbt.process_0/d/b4349aaadb7dfa45c3a53fc67c04b8f6",
    "ocdbt.process_0/d/caf01a82962cbd4651563d2ac1063e0b",
    "ocdbt.process_0/d/deefd3c43390a50472cbcd317b0fff58",
    "ocdbt.process_0/d/ec484cf8f02dcf59e1892180f0862e40",
    "ocdbt.process_0/manifest.ocdbt",
]

def check_aria2c():
    try:
        subprocess.run(["aria2c", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def download_file_python(url, local_path):
    print(f"Downloading {url} -> {local_path} ...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Simple chunked download with urllib
    try:
        with urllib.request.urlopen(url) as response, open(local_path, 'wb') as out_file:
            # Try to get size
            length = response.getheader('content-length')
            if length:
                length = int(length)
                blocksize = max(4096, length // 100)
            else:
                blocksize = 1024 * 1024
            
            downloaded = 0
            while True:
                buf = response.read(blocksize)
                if not buf:
                    break
                out_file.write(buf)
                downloaded += len(buf)
                if length:
                    percent = (downloaded / length) * 100
                    # Limit output spam
                    if downloaded % (blocksize * 5) == 0 or downloaded == length:
                        print(f"  {local_path.name}: {percent:.1f}% ({downloaded}/{length} bytes)")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False

def main():
    checkpoint_rel = DEFAULT_CHECKPOINT
    openpi_data_home = os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")
    target_dir = pathlib.Path(openpi_data_home).expanduser().resolve() / "openpi-assets" / checkpoint_rel
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Target directory: {target_dir}")
    print(f"Base GCS URL: {GCS_BASE_URL}/{checkpoint_rel}")
    
    # We will download the files using aria2c if available, otherwise python fallback
    has_aria = check_aria2c()
    
    if has_aria:
        print("Detected aria2c! Generating download input file and starting fast parallel download...")
        input_file_path = target_dir.parent / "aria2_input.txt"
        
        with open(input_file_path, "w") as f:
            for rel_file in PI05_BASE_FILES:
                url = f"{GCS_BASE_URL}/{checkpoint_rel}/{rel_file}"
                local_file = target_dir / rel_file
                # aria2c input format:
                # URL
                #   dir=DIRECTORY
                #   out=FILENAME
                f.write(f"{url}\n")
                f.write(f"  dir={local_file.parent}\n")
                f.write(f"  out={local_file.name}\n")
        
        print(f"Aria2c input file written to: {input_file_path}")
        # Run aria2c
        # -j 4: max 4 parallel downloads
        # -x 16: max 16 connections per server
        # -s 16: use 16 connections to download a single file
        # -c: continue downloading a partially-downloaded file (breakpoint resume!)
        cmd = [
            "aria2c",
            "-i", str(input_file_path),
            "-j", "4",
            "-x", "16",
            "-s", "16",
            "-c",
            "--summary-interval=10",
            "--file-allocation=none"
        ]
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("\nDownload completed successfully!")
            # Clean up the input file
            if input_file_path.exists():
                input_file_path.unlink()
        except subprocess.CalledProcessError as e:
            print(f"\naria2c failed with exit code {e.returncode}. Please resolve and try again.", file=sys.stderr)
            sys.exit(e.returncode)
    else:
        print("aria2c is NOT installed. Falling back to multi-threaded Python downloader.")
        print("Note: Python downloader is slower and does NOT support breakpoint resume.")
        print("TIP: For maximum download speed, install aria2: 'apt-get update && apt-get install -y aria2'")
        
        # Fallback to python thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for rel_file in PI05_BASE_FILES:
                url = f"{GCS_BASE_URL}/{checkpoint_rel}/{rel_file}"
                local_file = target_dir / rel_file
                futures.append(executor.submit(download_file_python, url, local_file))
            
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            if all(results):
                print("\nAll files downloaded successfully via Python fallback!")
            else:
                print("\nSome downloads failed. Please try running the script again.", file=sys.stderr)
                sys.exit(1)

if __name__ == "__main__":
    main()
