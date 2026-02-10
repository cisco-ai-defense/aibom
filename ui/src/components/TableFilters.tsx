// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

import { Search, Filter, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { Component } from '@/types/component';

interface TableFiltersProps {
  searchTerm: string;
  onSearchChange: (value: string) => void;
  selectedType: string;
  onTypeChange: (value: string) => void;
  onClearFilters: () => void;
  totalResults: number;
  filteredResults: number;
  availableTypes: string[];
}

export function TableFilters({
  searchTerm,
  onSearchChange,
  selectedType,
  onTypeChange,
  onClearFilters,
  totalResults,
  filteredResults,
  availableTypes,
}: TableFiltersProps) {
  const hasActiveFilters = searchTerm || selectedType !== 'all-types';

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'prompt': return 'Prompt';
      case 'retriever': return 'Retriever';
      case 'datastore': return 'Datastore';
      case 'tool': return 'Tool';
      case 'memory': return 'Memory';
      case 'agent': return 'Agent';
      default: return type.charAt(0).toUpperCase() + type.slice(1);
    }
  };

  return (
    <div className="space-y-4">
      {/* Search and Quick Stats */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search by component name or file path..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full sm:max-w-sm pl-10 bg-input border-border text-foreground placeholder:text-muted-foreground"
          />
        </div>
        
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>Showing {filteredResults} of {totalResults} components</span>
        </div>
      </div>

      {/* Filter Dropdowns */}
      <div className="flex flex-col sm:flex-row gap-4">
        <Select value={selectedType} onValueChange={onTypeChange}>
          <SelectTrigger className="w-full sm:w-[180px] bg-card border-border text-foreground">
            <SelectValue placeholder="Filter by type" />
          </SelectTrigger>
          <SelectContent className="bg-card border-border z-50 shadow-lg">
            <SelectItem value="all-types">All Components</SelectItem>
            {availableTypes.map((type) => (
              <SelectItem key={type} value={type}>
                {getTypeLabel(type)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Active Filter Tags */}
      {hasActiveFilters && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">Active filters:</span>
          {searchTerm && (
            <Badge variant="secondary" className="text-xs">
              Search: "{searchTerm}"
            </Badge>
          )}
          {selectedType && selectedType !== 'all-types' && (
            <Badge variant="secondary" className="text-xs">
              Type: {getTypeLabel(selectedType)}
            </Badge>
          )}
          <Button
            onClick={onClearFilters}
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs text-muted-foreground hover:text-foreground"
          >
            <X className="h-3 w-3 mr-1" />
            Clear filters
          </Button>
        </div>
      )}
    </div>
  );
}