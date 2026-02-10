// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function truncateMiddle(str: string, maxLength: number = 50): string {
  if (str.length <= maxLength) return str;
  
  const charsToShow = maxLength - 3; // Reserve 3 chars for "..."
  const frontChars = Math.ceil(charsToShow / 2);
  const backChars = Math.floor(charsToShow / 2);
  
  return str.substr(0, frontChars) + '...' + str.substr(str.length - backChars);
}
