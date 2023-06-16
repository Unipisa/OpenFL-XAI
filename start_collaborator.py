# Copyright (C) 2023 AI&RD Research Group, Department of Information Engineering, University of Pisa
# SPDX-License-Identifier: Apache-2.0

import os

collab_name = os.environ['COLLABORATOR_NAME']
os.system(f"fx collaborator start -n {collab_name}")
