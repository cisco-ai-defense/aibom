#!/usr/bin/env node
// Copyright 2026 Cisco Systems, Inc. and its affiliates
// SPDX-License-Identifier: Apache-2.0

"use strict";

const fs = require("fs");
const path = require("path");
const parser = require("@babel/parser");
const traverse = require("@babel/traverse").default;
const t = require("@babel/types");

function parseFile(filePath) {
  const source = fs.readFileSync(filePath, "utf-8");
  const ext = path.extname(filePath).toLowerCase();
  const isTS = ext === ".ts" || ext === ".tsx";
  const isJSX = ext === ".jsx" || ext === ".tsx";

  const plugins = ["decorators-legacy", "classProperties", "dynamicImport"];
  if (isTS) plugins.push("typescript");
  if (isJSX || !isTS) plugins.push("jsx");

  let ast;
  try {
    ast = parser.parse(source, {
      sourceType: "module",
      allowImportExportEverywhere: true,
      plugins,
    });
  } catch (_e) {
    return emptyResult(filePath);
  }

  const result = {
    file_path: filePath,
    assignments: [],
    calls: [],
    decorators: [],
    type_annotations: [],
    context_managers: [],
    class_defs: [],
    function_annotations: [],
    imports: [],
  };

  const importMap = {};

  traverse(ast, {
    ImportDeclaration(nodePath) {
      const src = nodePath.node.source.value;
      for (const spec of nodePath.node.specifiers) {
        if (t.isImportDefaultSpecifier(spec)) {
          importMap[spec.local.name] = `${src}.default`;
          result.imports.push(`from ${src} import ${spec.local.name}`);
        } else if (t.isImportSpecifier(spec)) {
          const imported = spec.imported.name || spec.imported.value;
          importMap[spec.local.name] = `${src}.${imported}`;
          if (spec.local.name !== imported) {
            result.imports.push(
              `from ${src} import ${imported} as ${spec.local.name}`
            );
          } else {
            result.imports.push(`from ${src} import ${imported}`);
          }
        } else if (t.isImportNamespaceSpecifier(spec)) {
          importMap[spec.local.name] = src;
          result.imports.push(`import ${src} as ${spec.local.name}`);
        }
      }
    },

    VariableDeclarator(nodePath) {
      const init = nodePath.node.init;
      if (!init) return;

      if (t.isCallExpression(init) && isRequire(init)) {
        handleRequire(init, nodePath.node.id, importMap, result);
        return;
      }

      const targetName = nodePath.node.id.name || "<destructured>";
      let emitted = false;

      const unwrapped = t.isAwaitExpression(init) ? init.argument : init;

      if (t.isNewExpression(unwrapped) || t.isCallExpression(unwrapped)) {
        const qName = resolveCallName(unwrapped, importMap);
        if (qName) {
          const args = extractArguments(unwrapped.arguments);
          const line = unwrapped.loc ? unwrapped.loc.start.line : 0;
          const rawCode = extractRawCode(source, unwrapped);
          result.assignments.push({
            target_qualified_name: targetName,
            call: {
              qualified_name: qName,
              arguments: args,
              line_number: line,
              raw_code: rawCode,
            },
            line_number: line,
          });
          emitted = true;
        }
      }

      if (t.isCallExpression(unwrapped) && !t.isNewExpression(unwrapped)) {
        const rootNew = findRootNewExpression(unwrapped);
        if (rootNew) {
          const qName = resolveCallName(rootNew, importMap);
          if (qName) {
            const args = extractArguments(rootNew.arguments);
            const line = rootNew.loc ? rootNew.loc.start.line : 0;
            const rawCode = extractRawCode(source, rootNew);
            result.assignments.push({
              target_qualified_name: targetName,
              call: {
                qualified_name: qName,
                arguments: args,
                line_number: line,
                raw_code: rawCode,
              },
              line_number: line,
            });
            emitted = true;
          }
        }
      }
    },

    ExpressionStatement(nodePath) {
      const expr = nodePath.node.expression;
      if (!t.isCallExpression(expr) && !t.isAwaitExpression(expr)) return;
      const call = t.isAwaitExpression(expr) ? expr.argument : expr;
      if (!t.isCallExpression(call)) return;
      const qName = resolveCallName(call, importMap);
      if (!qName) return;
      const args = extractArguments(call.arguments);
      const line = call.loc ? call.loc.start.line : 0;
      result.calls.push({
        qualified_name: qName,
        arguments: args,
        line_number: line,
        raw_code: extractRawCode(source, call),
      });
    },

    CallExpression(nodePath) {
      const parentType = nodePath.parent.type;
      if (
        parentType === "ExpressionStatement" ||
        parentType === "VariableDeclarator" ||
        parentType === "AwaitExpression"
      ) {
        return;
      }
      const node = nodePath.node;
      const qName = resolveCallName(node, importMap);
      if (!qName) return;
      const args = extractArguments(node.arguments);
      const line = node.loc ? node.loc.start.line : 0;
      result.calls.push({
        qualified_name: qName,
        arguments: args,
        line_number: line,
        raw_code: extractRawCode(source, node),
      });
    },

    NewExpression(nodePath) {
      const parentType = nodePath.parent.type;
      if (parentType === "VariableDeclarator") return;
      const node = nodePath.node;
      const qName = resolveCallName(node, importMap);
      if (!qName) return;
      const args = extractArguments(node.arguments);
      const line = node.loc ? node.loc.start.line : 0;
      result.calls.push({
        qualified_name: qName,
        arguments: args,
        line_number: line,
        raw_code: extractRawCode(source, node),
      });
    },

    ClassDeclaration(nodePath) {
      const node = nodePath.node;
      const name = node.id ? node.id.name : "<anonymous>";
      const bases = [];
      if (node.superClass) {
        const baseName = resolveExprName(node.superClass, importMap);
        if (baseName) bases.push(baseName);
      }
      const line = node.loc ? node.loc.start.line : 0;

      const decoratorObs = extractDecorators(node.decorators, importMap, source);
      for (const dec of decoratorObs) {
        dec.decorated_function_name = name;
        dec.line_number = line;
        result.decorators.push(dec);
      }

      result.class_defs.push({
        class_name: name,
        qualified_name: importMap[name] || name,
        base_classes: bases,
        line_number: line,
        aibom_annotation: null,
      });
    },

    ClassMethod(nodePath) {
      const node = nodePath.node;
      const decoratorObs = extractDecorators(node.decorators, importMap, source);
      const methodName = node.key.name || node.key.value || "<method>";
      for (const dec of decoratorObs) {
        dec.decorated_function_name = methodName;
        result.decorators.push(dec);
      }
    },
  });

  return result;
}

function emptyResult(filePath) {
  return {
    file_path: filePath,
    assignments: [],
    calls: [],
    decorators: [],
    type_annotations: [],
    context_managers: [],
    class_defs: [],
    function_annotations: [],
    imports: [],
  };
}

function isRequire(node) {
  return (
    t.isCallExpression(node) &&
    t.isIdentifier(node.callee, { name: "require" }) &&
    node.arguments.length === 1 &&
    t.isStringLiteral(node.arguments[0])
  );
}

function handleRequire(callNode, idNode, importMap, result) {
  const src = callNode.arguments[0].value;
  if (t.isIdentifier(idNode)) {
    importMap[idNode.name] = src;
    result.imports.push(`import ${src} as ${idNode.name}`);
  } else if (t.isObjectPattern(idNode)) {
    for (const prop of idNode.properties) {
      if (t.isObjectProperty(prop) && t.isIdentifier(prop.key)) {
        const localName = t.isIdentifier(prop.value)
          ? prop.value.name
          : prop.key.name;
        importMap[localName] = `${src}.${prop.key.name}`;
        if (localName !== prop.key.name) {
          result.imports.push(
            `from ${src} import ${prop.key.name} as ${localName}`
          );
        } else {
          result.imports.push(`from ${src} import ${prop.key.name}`);
        }
      }
    }
  }
}

function findRootNewExpression(node) {
  let current = node;
  while (t.isCallExpression(current)) {
    if (t.isMemberExpression(current.callee)) {
      current = current.callee.object;
    } else {
      break;
    }
  }
  return t.isNewExpression(current) ? current : null;
}

function resolveCallName(node, importMap) {
  const callee = t.isNewExpression(node) ? node : node;
  return resolveExprName(callee.callee, importMap);
}

function resolveExprName(node, importMap) {
  if (t.isIdentifier(node)) {
    return importMap[node.name] || node.name;
  }
  if (t.isMemberExpression(node)) {
    const parts = [];
    let current = node;
    while (t.isMemberExpression(current)) {
      const prop = current.property;
      parts.unshift(prop.name || prop.value || "");
      current = current.object;
    }
    if (t.isIdentifier(current)) {
      const rootResolved = importMap[current.name] || current.name;
      return rootResolved + "." + parts.join(".");
    }
  }
  return null;
}

function extractArguments(args) {
  const result = {};
  if (!args || args.length === 0) return result;
  for (const arg of args) {
    if (t.isObjectExpression(arg)) {
      for (const prop of arg.properties) {
        if (t.isObjectProperty(prop) && t.isIdentifier(prop.key)) {
          result[prop.key.name] = extractLiteralValue(prop.value);
        }
      }
    }
  }
  return result;
}

function extractLiteralValue(node) {
  if (t.isStringLiteral(node)) return node.value;
  if (t.isNumericLiteral(node)) return node.value;
  if (t.isBooleanLiteral(node)) return node.value;
  if (t.isNullLiteral(node)) return null;
  if (t.isArrayExpression(node)) {
    return node.elements.map((el) => (el ? extractLiteralValue(el) : null));
  }
  if (t.isIdentifier(node)) return `VARIABLE:${node.name}`;
  if (t.isTemplateLiteral(node)) {
    return node.quasis.map((q) => q.value.raw).join("${...}");
  }
  if (t.isObjectExpression(node)) {
    const obj = {};
    for (const prop of node.properties) {
      if (t.isObjectProperty(prop) && (t.isIdentifier(prop.key) || t.isStringLiteral(prop.key))) {
        const key = t.isIdentifier(prop.key) ? prop.key.name : prop.key.value;
        obj[key] = extractLiteralValue(prop.value);
      }
    }
    return obj;
  }
  if (t.isCallExpression(node) || t.isNewExpression(node)) {
    const callee = node.callee;
    if (t.isIdentifier(callee)) return `VARIABLE:${callee.name}`;
    if (t.isMemberExpression(callee) && t.isIdentifier(callee.property)) {
      return `VARIABLE:${callee.property.name}`;
    }
  }
  if (t.isAwaitExpression(node)) return extractLiteralValue(node.argument);
  return "<expression>";
}

function extractDecorators(decorators, importMap, _source) {
  if (!decorators || decorators.length === 0) return [];
  const results = [];
  for (const dec of decorators) {
    const expr = dec.expression;
    let qName;
    if (t.isCallExpression(expr)) {
      qName = resolveExprName(expr.callee, importMap);
    } else {
      qName = resolveExprName(expr, importMap);
    }
    if (qName) {
      const line = dec.loc ? dec.loc.start.line : 0;
      results.push({
        decorator_qualified_name: qName,
        decorated_function_name: "",
        line_number: line,
        instance_variable: null,
      });
    }
  }
  return results;
}

function extractRawCode(source, node) {
  if (!node.start || !node.end) return "";
  const snippet = source.slice(node.start, Math.min(node.end, node.start + 500));
  return snippet;
}

const filePath = process.argv[2];
if (!filePath) {
  process.stderr.write("Usage: parse.js <file_path>\n");
  process.exit(1);
}

try {
  const result = parseFile(filePath);
  process.stdout.write(JSON.stringify(result));
} catch (err) {
  process.stderr.write(`Parse error: ${err.message}\n`);
  process.stdout.write(JSON.stringify(emptyResult(filePath)));
}
