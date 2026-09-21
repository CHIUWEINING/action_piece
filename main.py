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

"""Main file for ActionPiece."""

import argparse

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', type=str, default='ActionPiece', help='Model name'
  )
  parser.add_argument(
      '--dataset', type=str, default='AmazonReviews2023', help='Dataset name'
  )
  return parser.parse_known_args()


if __name__ == '__main__':
  args, unparsed_args = parse_args()
  command_line_configs = parse_command_line_args(unparsed_args)

  pipeline = Pipeline(
      model_name=args.model,
      dataset_name=args.dataset,
      config_dict=command_line_configs,
  )
  pipeline.run()
