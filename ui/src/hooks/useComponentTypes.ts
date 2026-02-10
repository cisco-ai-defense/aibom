// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';

interface ComponentTypesResponse {
  types: string[];
}

export const useComponentTypes = () => {
  return useQuery({
    queryKey: ['component-types'],
    queryFn: async (): Promise<string[]> => {
      const res = await api.get('/api/components/types');
      const data: ComponentTypesResponse = res.data;
      return data.types;
    },
    staleTime: 5 * 60 * 1000,
  });
};
