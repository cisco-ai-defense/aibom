// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

import { http, HttpResponse } from "msw";
import { sampleComponents } from "./sampleData";
import type { Component } from "@/types/component";

const componentsData = {
  components: sampleComponents,
  total: sampleComponents.length,
};

export const handlers = [
  // GET /health - Health check
  http.get("/health", () => {
    return HttpResponse.json({
      status: "healthy",
      total_components: sampleComponents.length,
    });
  }),

  // GET /api/components - Return all components (flat array)
  http.get("/api/components", ({ request }) => {
    const url = new URL(request.url);
    const typeFilter = url.searchParams.get("type");
    const filePathFilter = url.searchParams.get("file_path");

    let filtered: Component[] = sampleComponents;
    if (typeFilter) {
      filtered = filtered.filter((c) => c.type === typeFilter);
    }
    if (filePathFilter) {
      const fp = filePathFilter.toLowerCase();
      filtered = filtered.filter((c) => c.file_path.toLowerCase().includes(fp));
    }

    return HttpResponse.json({ components: filtered, total: filtered.length });
  }),

  // GET /api/components/types - Return available component categories
  http.get("/api/components/types", () => {
    const types = Array.from(new Set(sampleComponents.map((c: Component) => c.type)));
    return HttpResponse.json({ types });
  }),

  // GET /api/components/:component_id - Return specific component
  http.get("/api/components/:component_id", ({ params }) => {
    const { component_id } = params as { component_id: string };
    const component = sampleComponents.find(
      (c: Component) => c.id === component_id
    );

    if (!component) {
      return new HttpResponse(null, { status: 404 });
    }

    return HttpResponse.json(component);
  }),
];
