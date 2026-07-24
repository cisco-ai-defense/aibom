# Package liveness freshness

When a selected knowledge base contains a `package_catalog`, AI BOM adds its
frozen liveness fields to dependency components that were already discovered
in the user's scan. It does not enumerate or probe packages that are absent
from the AI BOM.

The component `metadata` object may contain:

- `liveness_status`
- `liveness_snapshot_at`
- `liveness_certification`
- `as_of` and `liveness_observed_at` when a newer live delta is returned

The frozen snapshot is always the fallback. If its timestamp is more than
seven days old and the manifest advertises `freshness_api`, the CLI may send
an anonymous request containing only the ecosystem and normalized name of the
packages already present in the scan. Requests contain at most 100 packages.
No tenant credential or authorization header is sent. Timeouts, rate limits,
invalid responses, and other endpoint failures leave the frozen snapshot in
place and do not fail the scan.

Use either of these options to prevent live freshness requests:

```bash
cisco-aibom analyze ./my-app --no-network ...
cisco-aibom analyze ./my-app --liveness-only-snapshot ...
```

For this feature, the options are equivalent: both retain local snapshot
fields and suppress the freshness request. They are also available through
`AIBOM_NO_NETWORK` and `AIBOM_LIVENESS_ONLY_SNAPSHOT`.

`CISCO_AIBOM_FRESHNESS_URL` can explicitly select a freshness endpoint for
testing or a custom knowledge-base distribution. Otherwise the CLI uses the
`freshness_api` value from the selected manifest. No default endpoint is
shipped.
