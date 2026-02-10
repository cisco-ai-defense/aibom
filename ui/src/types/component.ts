// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

export interface Component {
  id: string;
  name: string;
  file_path: string;
  line_number: number;
  type: 'prompt' | 'retriever' | 'datastore' | 'tool' | 'memory' | 'agent' | 'other' | string;
  text: string | null;
  model_name: string | null;
  embedding_model: string | null;
  additional_data?: Record<string, unknown> | null;
}
