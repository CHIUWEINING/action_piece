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

"""Datasets loaded from a prebuilt GenRec cache (Steam / Yelp).

The steam-yelp-datasets folder ships the GenRec cache that was used for the
other baselines (`<Dataset>/genrec-processed/`):
  - all_item_seqs.json      {user: [item, ...]}, full time-ordered sequences
  - id_mapping.json         user2id / item2id / id2user / id2item (0 = [PAD])
  - metadata.sentence.json  {item: text}, the text fed to the sentence encoder
These files are read as is, so ActionPiece uses exactly the same sequences,
leave-one-out split and item text as the other baselines. The folder itself is
never written to: tokenizer outputs (sentence embeddings, semantic IDs,
actionpiece.json) go to `<cache_dir>/<Dataset>/processed/`.
"""

import hashlib
import json
import os
from typing import Any

from genrec.dataset import AbstractDataset

PREBUILT_FILES = (
    'all_item_seqs.json',
    'id_mapping.json',
    'metadata.sentence.json',
)


def sha256sum(path: str) -> str:
  """Returns the sha256 hex digest of a file."""
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


class PrebuiltGenRecDataset(AbstractDataset):
  """Base class for datasets whose GenRec cache is already built.

  Subclasses only set `name`, the folder name under `data_dir`.

  Attributes:
      data_dir (str): The steam-yelp-datasets folder (config `data_dir`).
      source_dir (str): `<data_dir>/<name>/genrec-processed`, read only.
      cache_dir (str): `<cache_dir>/<name>`; the tokenizer writes to its
        `processed/` subfolder.
  """

  name = None

  def __init__(self, config: dict[str, Any]):
    super().__init__(config)
    # The logger / trainer already use config['category'] for their paths, so
    # a different value cannot be fixed here.
    assert config.get('category') == self.name, (
        f'{self.name} has no categories: drop --category or set'
        f' --category={self.name} (got {config.get("category")!r}).'
    )
    self.category = self.name
    self.log(f'[DATASET] {self.name} (prebuilt GenRec cache)')

    self.data_dir = config['data_dir']
    self.source_dir = os.path.join(self.data_dir, self.name, 'genrec-processed')
    self.cache_dir = os.path.join(config['cache_dir'], self.name)
    os.makedirs(os.path.join(self.cache_dir, 'processed'), exist_ok=True)
    self._load_prebuilt()

  def _verify_checksums(self):
    """Checks the prebuilt files against <data_dir>/SHA256SUMS."""
    sums_file = os.path.join(self.data_dir, 'SHA256SUMS')
    if not os.path.exists(sums_file):
      self.log(
          f'[DATASET] {sums_file} not found, skipping checksum check.',
          level='warning',
      )
      return
    expected = {}
    with open(sums_file, 'r') as f:
      for line in f:
        if line.strip():
          digest, rel_path = line.split(maxsplit=1)
          expected[rel_path.strip()] = digest
    for file_name in PREBUILT_FILES:
      rel_path = f'{self.name}/genrec-processed/{file_name}'
      assert rel_path in expected, f'{rel_path} is not listed in {sums_file}'
      actual = sha256sum(os.path.join(self.data_dir, rel_path))
      assert actual == expected[rel_path], (
          f'Checksum mismatch for {rel_path}: the file differs from the one'
          ' used by the other baselines.'
      )

  def _load_prebuilt(self):
    """Loads all_item_seqs, id_mapping and item2meta from source_dir."""
    for file_name in PREBUILT_FILES:
      path = os.path.join(self.source_dir, file_name)
      assert os.path.exists(path), (
          f'{path} not found. Set --data_dir to the steam-yelp-datasets folder.'
      )
    self._verify_checksums()

    with open(os.path.join(self.source_dir, 'all_item_seqs.json'), 'r') as f:
      self.all_item_seqs = json.load(f)
    with open(os.path.join(self.source_dir, 'id_mapping.json'), 'r') as f:
      self.id_mapping = json.load(f)

    if self.config['metadata'] == 'none':
      self.item2meta = None
    elif self.config['metadata'] == 'sentence':
      with open(
          os.path.join(self.source_dir, 'metadata.sentence.json'), 'r'
      ) as f:
        self.item2meta = json.load(f)
    else:
      raise NotImplementedError(
          f'metadata={self.config["metadata"]} is not available for'
          f' {self.name} (only "sentence" / "none").'
      )
