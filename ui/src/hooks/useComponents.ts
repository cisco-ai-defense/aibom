// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Component } from '@/types/component';

interface ComponentsResponse {
  components: Component[];
  total: number;
}

export const useComponents = () => {
  return useQuery({
    queryKey: ['components'],
    queryFn: async (): Promise<Component[]> => {
      const response = await api.get('/api/components');
      const data: ComponentsResponse = response.data;
      return data.components;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useComponent = (id: string) => {
  return useQuery({
    queryKey: ['component', id],
    queryFn: async (): Promise<Component> => {
      const response = await api.get(`/api/components/${id}`);
      return response.data as Component;
    },
    enabled: !!id,
  });
};