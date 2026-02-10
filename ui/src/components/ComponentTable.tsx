// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

import { Code, FileText } from 'lucide-react';
import { truncateMiddle } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card, CardContent } from '@/components/ui/card';
import type { Component } from '@/types/component';

interface ComponentTableProps {
  components: Component[];
  selectedComponents: string[];
  onSelectComponent: (componentId: string) => void;
  onSelectAll: (checked: boolean) => void;
}

export function ComponentTable({
  components,
  selectedComponents,
  onSelectComponent,
  onSelectAll,
}: ComponentTableProps) {
  const allSelected = components.length > 0 && selectedComponents.length === components.length;
  const someSelected = selectedComponents.length > 0 && selectedComponents.length < components.length;

  return (
    <Card className="border-border bg-card">
      <CardContent className="p-0">
        {components.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No components found matching the current filters.</p>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-muted/30">
                <TableHead className="w-12">
                  <Checkbox
                    checked={allSelected}
                    onCheckedChange={onSelectAll}
                    aria-label="Select all"
                    className="border-border data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                    ref={(el) => {
                      const input = el as unknown as HTMLInputElement | null;
                      if (input && 'indeterminate' in input) {
                        input.indeterminate = someSelected;
                      }
                    }}
                  />
                </TableHead>
                <TableHead className="text-muted-foreground font-medium">
                  Component Name
                </TableHead>
                <TableHead className="text-muted-foreground font-medium">
                  File Path
                </TableHead>
                <TableHead className="text-muted-foreground font-medium">Line Number</TableHead>
                <TableHead className="text-muted-foreground font-medium">Category</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {components.map((component) => (
                <TableRow 
                  key={component.id} 
                  className="border-border hover:bg-muted/50 transition-colors"
                  data-state={selectedComponents.includes(component.id) ? "selected" : undefined}
                >
                  <TableCell>
                    <Checkbox
                      checked={selectedComponents.includes(component.id)}
                      onCheckedChange={() => onSelectComponent(component.id)}
                      aria-label={`Select ${component.name}`}
                      className="border-border data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                    />
                  </TableCell>
                  <TableCell className="font-medium text-foreground">
                    <div className="flex items-center space-x-3 min-w-0">
                      <Code className="h-4 w-4 text-primary" />
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="font-mono text-sm truncate max-w-[300px] cursor-help">
                            {component.name}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent side="top" align="start">
                          <span className="font-mono text-xs break-all">{component.name}</span>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    <div className="flex items-center space-x-2">
                      <FileText className="h-4 w-4" />
                      <span className="font-mono text-xs max-w-[300px]" title={component.file_path}>
                        {truncateMiddle(component.file_path, 50)}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground font-mono">{component.line_number}</TableCell>
                  <TableCell>
                    <Badge
                      variant="secondary"
                      className={
                        component.type === 'prompt' ? 'bg-green-500 text-white' :
                        component.type === 'retriever' ? 'bg-blue-500 text-white' :
                        component.type === 'datastore' ? 'bg-primary text-primary-foreground' :
                        component.type === 'tool' ? 'bg-amber-500 text-white' :
                        component.type === 'memory' ? 'bg-purple-600 text-white' :
                        component.type === 'agent' ? 'bg-pink-600 text-white' :
                        'bg-muted text-muted-foreground'
                      }
                    >
                      {component.type.charAt(0).toUpperCase() + component.type.slice(1)}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}