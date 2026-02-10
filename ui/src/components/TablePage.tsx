// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

import { useState, useMemo } from 'react';
import { Code, Download } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { TableFilters } from './TableFilters';
import { ComponentTable } from './ComponentTable';
import { useComponents } from '@/hooks/useComponents';
import { useComponentTypes } from '@/hooks/useComponentTypes';
import type { Component } from '@/types/component';

export function TablePage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState('all-types');
  const [selectedComponents, setSelectedComponents] = useState<string[]>([]);
  const { data: components = [], isLoading, isError } = useComponents();
  const { data: types = [] } = useComponentTypes();

  // Filter components based on search and type
  const filteredComponents = useMemo(() => {
    return components.filter((component: Component) => {
      const matchesSearch = !searchTerm || 
        component.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        component.file_path.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesType = selectedType === 'all-types' || component.type === selectedType;

      return matchesSearch && matchesType;
    });
  }, [components, searchTerm, selectedType]);

  const handleSelectComponent = (componentId: string) => {
    setSelectedComponents(prev => 
      prev.includes(componentId) 
        ? prev.filter(id => id !== componentId)
        : [...prev, componentId]
    );
  };

  const handleSelectAll = (checked: boolean) => {
    setSelectedComponents(checked ? filteredComponents.map(component => component.id) : []);
  };

  const handleClearFilters = () => {
    setSearchTerm('');
    setSelectedType('all-types');
  };

  const handleExport = () => {
    const dataToExport = filteredComponents.map(component => ({
      name: component.name,
      type: component.type,
      file_path: component.file_path,
      line_number: component.line_number
    }));

    const jsonStr = JSON.stringify(dataToExport, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `components-export-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const stats = useMemo(() => {
    const total = components.length;
    const perType: Record<string, number> = {};
    types.forEach((t) => {
      perType[t] = components.filter((c) => c.type === t).length;
    });
    return { total, perType } as const;
  }, [components, types]);

  const availableTypes = useMemo(() => {
    return (types && types.length > 0
      ? [...types]
      : [...new Set(components.map(component => component.type))]
    ).sort();
  }, [types, components]);

  // Top 5 categories by count for stats cards
  const topTypes = useMemo(() => {
    return Object.entries(stats.perType)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([t]) => t);
  }, [stats.perType]);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Code className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-3xl font-bold text-foreground">Component Analysis</h1>
                <p className="text-muted-foreground">View and analyze MCP components</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button onClick={handleExport} variant="outline" className="gap-2">
              <Download className="h-4 w-4" />
              Export JSON
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4 mb-8">
          <Card className="border-border bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Components</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{stats.total}</div>
            </CardContent>
          </Card>
          
          {topTypes.map((t) => (
            <Card key={t} className="border-border bg-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-foreground">{stats.perType[t] ?? 0}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Loading / Error States */}
        {isLoading && (
          <div className="text-sm text-muted-foreground">Loading components…</div>
        )}
        {isError && (
          <div className="text-sm text-red-600">Failed to load components.</div>
        )}

        {/* Filters */}
        <Card className="border-border bg-card mb-6">
          <CardContent className="pt-6">
            <TableFilters
              searchTerm={searchTerm}
              onSearchChange={setSearchTerm}
              selectedType={selectedType}
              onTypeChange={setSelectedType}
              onClearFilters={handleClearFilters}
              totalResults={components.length}
              filteredResults={filteredComponents.length}
              availableTypes={availableTypes}
            />
          </CardContent>
        </Card>

        {/* Table */}
        <ComponentTable
          components={filteredComponents}
          selectedComponents={selectedComponents}
          onSelectComponent={handleSelectComponent}
          onSelectAll={handleSelectAll}
        />
      </div>
    </div>
  );
}