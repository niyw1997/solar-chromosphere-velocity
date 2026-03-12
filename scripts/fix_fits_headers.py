#!/usr/bin/env python3
"""
Script to fix corrupted FITS headers containing non-standard ASCII characters.
"""

import os
import sys
from astropy.io import fits
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_fits_header(header):
    """Clean FITS header by removing non-standard ASCII characters."""
    cleaned = fits.Header()
    
    for card in header.cards:
        try:
            key = card.keyword
            value = card.value
            comment = card.comment
            
            # Clean string values
            if isinstance(value, str):
                # Replace newlines with spaces
                cleaned_value = value.replace('\n', ' ').replace('\r', ' ')
                # Remove other non-printable ASCII characters
                cleaned_value = ''.join(char for char in cleaned_value if 32 <= ord(char) <= 126)
                # Truncate if too long
                if len(cleaned_value) > 70:
                    cleaned_value = cleaned_value[:70]
                    logger.debug(f"Truncated value for key {key}")
                
                # Clean comment if it's a string
                if isinstance(comment, str):
                    cleaned_comment = comment.replace('\n', ' ').replace('\r', ' ')
                    cleaned_comment = ''.join(char for char in cleaned_comment if 32 <= ord(char) <= 126)
                    if len(cleaned_comment) > 70:
                        cleaned_comment = cleaned_comment[:70]
                else:
                    cleaned_comment = comment
                
                cleaned[key] = (cleaned_value, cleaned_comment)
            else:
                # Non-string values can be added as-is
                cleaned.append(card)
                
        except Exception as e:
            logger.warning(f"Failed to clean card {card.keyword}: {e}")
            # Skip problematic cards
    
    return cleaned


def fix_fits_file(input_path, output_path=None, backup=True):
    """Fix a single FITS file with corrupted headers."""
    if output_path is None:
        output_path = input_path
    
    logger.info(f"Processing: {input_path}")
    
    try:
        # Read the file with error handling
        with fits.open(input_path, fix=True, ignore_missing_end=True) as hdul:
            # Create a new HDU list with cleaned headers
            new_hdul = fits.HDUList()
            
            for i, hdu in enumerate(hdul):
                if i == 0:
                    # Primary HDU
                    cleaned_header = clean_fits_header(hdu.header)
                    new_hdu = fits.PrimaryHDU(hdu.data, header=cleaned_header)
                else:
                    # Extension HDU
                    cleaned_header = clean_fits_header(hdu.header)
                    new_hdu = fits.ImageHDU(hdu.data, header=cleaned_header)
                
                new_hdul.append(new_hdu)
            
            # Backup original if requested
            if backup and input_path == output_path:
                backup_path = str(Path(input_path).with_suffix('.fits.bak'))
                import shutil
                shutil.copy2(input_path, backup_path)
                logger.info(f"Backup created: {backup_path}")
            
            # Save the cleaned file
            new_hdul.writeto(output_path, overwrite=True)
            logger.info(f"Saved cleaned file: {output_path}")
            
            return True
            
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return False


def fix_directory(input_dir, output_dir=None, pattern="*.fits", recursive=False):
    """Fix all FITS files in a directory."""
    from pathlib import Path
    import glob
    
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find files
    if recursive:
        files = list(input_path.rglob(pattern))
    else:
        files = list(input_path.glob(pattern))
    
    logger.info(f"Found {len(files)} FITS files to process")
    
    success_count = 0
    for filepath in files:
        if output_dir == input_dir:
            output_file = filepath
        else:
            output_file = output_path / filepath.relative_to(input_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if fix_fits_file(str(filepath), str(output_file)):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(files)} files")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Fix FITS files with corrupted headers")
    parser.add_argument("input", help="Input FITS file or directory")
    parser.add_argument("--output", "-o", help="Output file or directory (default: overwrite input)")
    parser.add_argument("--pattern", "-p", default="*.fits", help="File pattern (default: *.fits)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backup files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        fix_fits_file(str(input_path), args.output, backup=not args.no_backup)
    elif input_path.is_dir():
        # Process directory
        fix_directory(str(input_path), args.output, args.pattern, args.recursive)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()