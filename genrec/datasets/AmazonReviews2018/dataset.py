# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dataset for Amazon Reviews 2018 (amazon_v2, McAuley Lab)."""

import collections
import gzip
import json
import os
from typing import Any, Optional, Sequence

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text
from genrec.utils import download_file
import numpy as np
import tqdm

BASE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2'

AVAILABLE_CATEGORIES = [
    'AMAZON_FASHION', 'All_Beauty', 'Appliances', 'Arts_Crafts_and_Sewing',
    'Automotive', 'Books', 'CDs_and_Vinyl', 'Cell_Phones_and_Accessories',
    'Clothing_Shoes_and_Jewelry', 'Digital_Music', 'Electronics',
    'Gift_Cards', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen',
    'Industrial_and_Scientific', 'Kindle_Store', 'Luxury_Beauty',
    'Magazine_Subscriptions', 'Movies_and_TV', 'Musical_Instruments',
    'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies',
    'Prime_Pantry', 'Software', 'Sports_and_Outdoors',
    'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games',
]


def check_available_category(category: str):
  assert category in AVAILABLE_CATEGORIES, (
      f'Category "{category}" not available in Amazon 2018. '
      f'Available categories: {AVAILABLE_CATEGORIES}'
  )


def parse_gz(path: str):
  """Yields one dict per line of a gzipped JSON-lines file.

  Both 2018 reviews and metadata are valid JSON; unparsable lines are skipped.
  """
  with gzip.open(path, 'rt', encoding='utf-8') as g:
    for line in g:
      line = line.strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        continue


def get_item_seqs(
    reviews: Sequence[tuple[str, str, int]]
) -> dict[str, list[str]]:
  """Group the reviews by user and sort the items by time.

  Args:
      reviews (Sequence[tuple[str, str, int]]): A list of tuples representing
      the reviews. Each tuple contains the user, item, and time.

  Returns:
      dict: A dictionary where the keys are the users and the values are lists
      of items sorted by time.
  """
  # Group reviews by user
  item_seqs = collections.defaultdict(list)
  for user, item, time in reviews:
    item_seqs[user].append((item, time))

  # Sort items by time
  for user, item_time in item_seqs.items():
    item_time.sort(key=lambda x: x[1])
    item_seqs[user] = [item for item, _ in item_time]

  return item_seqs


class AmazonReviews2018(AbstractDataset):
  """Amazon Reviews 2018 dataset (https://nijianmo.github.io/amazon/)."""

  def __init__(self, config: dict[str, Any]):
    super().__init__(config)

    self.category = config['category']
    check_available_category(self.category)
    self.log(f'[DATASET] Amazon Reviews 2018 for category: {self.category}')

    self.cache_dir = os.path.join(
        config['cache_dir'], 'AmazonReviews2018', self.category
    )
    self._download_and_process_raw()

  def _download_raw(self, path: str, file_type: str = 'reviews') -> str:
    """Downloads the 5-core reviews file or the metadata file of the category.

    Skips the download if the file already exists under `path`.
    """
    if file_type == 'reviews':
      url = f'{BASE_URL}/categoryFilesSmall/{self.category}_5.json.gz'
    elif file_type == 'meta':
      url = f'{BASE_URL}/metaFiles2/meta_{self.category}.json.gz'
    else:
      raise ValueError(f'Unknown file type: {file_type}')
    local_filepath = os.path.join(path, os.path.basename(url))
    if not os.path.exists(local_filepath):
      download_file(url, local_filepath)
    assert os.path.exists(local_filepath), (
        f'Failed to download {url}. Please download it manually to'
        f' {local_filepath}.'
    )
    return local_filepath

  def _load_reviews(self, path: str) -> list[tuple[str, str, int]]:
    """Loads (user, item, timestamp) from the 5-core reviews json.gz."""
    self.log('[DATASET] Loading reviews...')
    reviews = []
    for inter in parse_gz(path):
      reviews.append(
          (inter['reviewerID'], inter['asin'], int(inter['unixReviewTime']))
      )
    return reviews

  def _remap_ids(
      self, item_seqs: dict[str, list[str]]
  ) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Builds user / item id mappings (id 0 is reserved for [PAD])."""
    self.log('[DATASET] Remapping user and item IDs...')
    for user, items in item_seqs.items():
      if user not in self.id_mapping['user2id']:
        self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
        self.id_mapping['id2user'].append(user)
      for item in items:
        if item not in self.id_mapping['item2id']:
          self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
          self.id_mapping['id2item'].append(item)
      self.all_item_seqs[user] = list(items)
    return self.all_item_seqs, self.id_mapping

  def _process_reviews(
      self, input_path: str, output_path: str
  ) -> tuple[dict[str, list[str]], dict[str, Any]]:
    seq_file = os.path.join(output_path, 'all_item_seqs.json')
    id_mapping_file = os.path.join(output_path, 'id_mapping.json')
    if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
      self.log('[DATASET] Reviews have been processed...')
      with open(seq_file, 'r') as f:
        all_item_seqs = json.load(f)
      with open(id_mapping_file, 'r') as f:
        id_mapping = json.load(f)
      return all_item_seqs, id_mapping

    self.log('[DATASET] Processing reviews...')
    reviews = self._load_reviews(input_path)
    item_seqs = get_item_seqs(reviews)
    all_item_seqs, id_mapping = self._remap_ids(item_seqs)

    self.log('[DATASET] Saving mapping data...')
    with open(seq_file, 'w') as f:
      json.dump(all_item_seqs, f)
    with open(id_mapping_file, 'w') as f:
      json.dump(id_mapping, f)
    return all_item_seqs, id_mapping

  def _load_metadata(
      self, path: str, item2id: dict[str, int]
  ) -> dict[str, Any]:
    """Loads metadata of items in item2id."""
    self.log('[DATASET] Loading metadata...')
    data = {}
    item_asins = set(item2id.keys())
    for info in tqdm.tqdm(parse_gz(path)):
      asin = info.get('asin')
      if asin in item_asins:
        data[asin] = info  # same as original 2014 code: last duplicate wins
    n_missing = len(item_asins - {'[PAD]'} - set(data))
    if n_missing:
      self.log(
          f'[DATASET] {n_missing} items have no metadata, using empty text.',
          level='warning',
      )
    return data

  def _sent_process(self, raw: Any) -> str:
    """Process the raw input according to the raw data type and return a processed sentence.

    The same implementation is used in AmazonReviews2014 / 2018 / 2023. It
    follows the original AmazonReviews2014 code, except that:
      - category names are no longer truncated by `clean_text(v)[:-1]`;
      - list[str] values are joined with ', ' (empty items dropped) and end
        with a period, instead of being concatenated without a separator;
      - None returns '' and int is handled like float.

    Args:
        raw (Any): The raw input to be processed (str, int, float, list[str],
          list[list[str]] or None).

    Returns:
        str: The processed sentence, followed by a space ('' for None).
    """
    if raw is None:
      return ''
    sentence = ''
    if isinstance(raw, (int, float)):
      sentence += str(raw)
      sentence += '.'
    elif raw and isinstance(raw[0], list):
      # list[list[str]], e.g. 2014 `categories`:
      # [['Beauty', 'Skin Care', 'Face'], ['Beauty', 'Makeup']]
      parts = [clean_text(v).strip() for v1 in raw for v in v1]
      sentence = ', '.join(p for p in parts if p).rstrip('.') + '.'
    elif isinstance(raw, list):
      # list[str], e.g. 2018 `feature` / `category`, 2023 `features` / `categories`
      parts = [clean_text(v).strip() for v in raw if isinstance(v, str)]
      parts = [p for p in parts if p]
      if parts:
        sentence = ', '.join(parts).rstrip('.') + '.'
    else:
      sentence = clean_text(raw)
    return sentence + ' '

  def _extract_meta_sentences(self, metadata: dict[str, Any]) -> dict[str, str]:
    """Builds one sentence per item from title / price / brand / feature / category / description.

    Items whose metadata is missing are filled with '' in `_process_meta`.
    """
    self.log('[DATASET] Extracting meta sentences...')
    item2meta = {}
    for item, meta in tqdm.tqdm(metadata.items()):
      meta_sentence = ''
      # Same fields / order as the original 2014 code (2018 renamed
      # `categories` -> `category`).
      for feature in [
          'title', 'price', 'brand', 'feature', 'category', 'description'
      ]:
        if feature in meta:
          meta_sentence += self._sent_process(meta[feature])
      item2meta[item] = meta_sentence
    return item2meta

  def _process_meta(
      self, input_path: str, output_path: str
  ) -> Optional[dict[str, Any]]:
    process_mode = self.config['metadata']
    meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')
    if os.path.exists(meta_file):
      self.log('[DATASET] Metadata has been processed...')
      with open(meta_file, 'r') as f:
        return json.load(f)

    self.log(f'[DATASET] Processing metadata, mode: {process_mode}')
    if process_mode == 'none':
      return None

    item2raw_meta = self._load_metadata(path=input_path, item2id=self.item2id)
    if process_mode == 'raw':
      item2meta = item2raw_meta
    elif process_mode == 'sentence':
      item2meta = self._extract_meta_sentences(metadata=item2raw_meta)
      for item in self.id_mapping['id2item'][1:]:
        item2meta.setdefault(item, '')
    else:
      raise NotImplementedError('Metadata processing type not implemented.')

    with open(meta_file, 'w') as f:
      json.dump(item2meta, f)
    return item2meta

  def _download_and_process_raw(self):
    raw_data_path = os.path.join(self.cache_dir, 'raw')
    os.makedirs(raw_data_path, exist_ok=True)
    with self.accelerator.main_process_first():  # only download once when ddp
      reviews_localpath = self._download_raw(
          path=raw_data_path, file_type='reviews'
      )
      meta_localpath = self._download_raw(path=raw_data_path, file_type='meta')

    np.random.seed(12345)

    processed_data_path = os.path.join(self.cache_dir, 'processed')
    os.makedirs(processed_data_path, exist_ok=True)

    self.all_item_seqs, self.id_mapping = self._process_reviews(
        input_path=reviews_localpath, output_path=processed_data_path
    )

    self.item2meta = self._process_meta(
        input_path=meta_localpath, output_path=processed_data_path
    )
