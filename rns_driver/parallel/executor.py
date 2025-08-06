# rns_driver/parallel/executor.py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Dict
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from ..config.settings import RNSConfig
from ..core.eos_catalog import EOSCatalog


class ParallelExecutor:
    """Manages parallel execution of RNS calculations."""
    
    def __init__(self, config: RNSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Determine number of workers
        if config.max_workers is None:
            self.n_workers = mp.cpu_count()
        else:
            self.n_workers = min(config.max_workers, mp.cpu_count())
    
    def process_eos_files(self, 
                     eos_args: List[Any],  # Changed from eos_files
                     processor: Callable[[Any], pd.DataFrame],
                     output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Process multiple EOS files in parallel.

        Args:
            eos_args: List of arguments for the processor function
            processor: Function that processes a single set of arguments
            output_dir: Optional directory to save intermediate results

        Returns:
            Combined DataFrame with all results
        """
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Split work among processes
        chunks = self._split_work(eos_args, self.n_workers)

        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, processor, output_dir): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results with progress bar
            with tqdm(total=len(eos_args), desc="Processing EOS files") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        results.append(chunk_result)
                        pbar.update(len(chunks[chunk_id]))

                        # Save intermediate result if requested
                        if output_dir and not chunk_result.empty:
                            chunk_path = output_dir / f"chunk_{chunk_id}.parquet"
                            chunk_result.to_parquet(chunk_path, engine='pyarrow')

                    except Exception as e:
                        self.logger.error(f"Chunk {chunk_id} failed: {e}")
                        pbar.update(len(chunks[chunk_id]))

        # Combine all results
        if results:
            combined_df = pd.concat(results, ignore_index=True)
            self.logger.info(f"Processed {len(combined_df)} neutron star models")
            return combined_df
        else:
            return pd.DataFrame()

    def _process_chunk(self, 
                      eos_args: List[Any],  # Changed from eos_files
                      processor: Callable[[Any], pd.DataFrame],
                      output_dir: Optional[Path] = None) -> pd.DataFrame:
        """Process a chunk of EOS arguments."""
        chunk_results = []
        
        for args in eos_args:  # Changed from eos_file
            try:
                result = processor(args)  # Pass args instead of eos_file
                if not result.empty:
                    chunk_results.append(result)
            except Exception as e:
                # Extract eos path for error logging
                eos_path = args[0] if isinstance(args, tuple) else args
                self.logger.error(f"Failed to process {eos_path}: {e}")
        
        if chunk_results:
            return pd.concat(chunk_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _split_work(self, items: List[Any], n_chunks: int) -> List[List[Any]]:
        """Split items into roughly equal chunks."""
        chunk_size = len(items) // n_chunks
        remainder = len(items) % n_chunks
        
        chunks = []
        start = 0
        
        for i in range(n_chunks):
            # Add 1 to chunk_size for the first 'remainder' chunks
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            if current_chunk_size > 0:
                chunks.append(items[start:start + current_chunk_size])
                start += current_chunk_size
        
        return chunks
    
    def map_parallel(self,
                    func: Callable,
                    items: List[Any],
                    desc: str = "Processing") -> List[Any]:
        """Generic parallel map function."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(func, item): item for item in items}
            
            with tqdm(total=len(items), desc=desc) as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Task failed: {e}")
                    pbar.update(1)
        
        return results