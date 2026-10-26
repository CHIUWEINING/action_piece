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

"""Yelp dataset, loaded from the prebuilt GenRec cache in steam-yelp-datasets."""

from genrec.datasets.prebuilt import PrebuiltGenRecDataset


class Yelp(PrebuiltGenRecDataset):
  """Yelp dataset. See `genrec/datasets/prebuilt.py` and steam-yelp-datasets/README.md."""

  name = 'Yelp'
