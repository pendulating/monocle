# Cyclomedia Street Smart API — Navigation & Recording Adjacency Reference

> Reverse-engineering reference for overhauling `urbanroamvqa` graph connectivity.
> Goal: replace spatial-buffer heuristics with the same signals the Street Smart viewer uses.

---

## How the Street Smart Viewer Navigates Between Recordings

The Street Smart JavaScript API does **not** read a pre-computed adjacency graph. Navigation arrows ("drive forward/backward") are resolved dynamically using three signals:

1. **Spatial proximity** — nearby recordings within ~5-10m capture intervals
2. **Capture timestamp (`recordedAt`)** — follow the same vehicle's sequential pass
3. **Viewing angle / recorder direction** — alignment between camera heading and bearing to candidate

The viewer queries the **Atlas WFS** for recordings in the current viewport, then selects the next recording by combining these three signals to follow the capture vehicle's trajectory.

---

## Atlas WFS Recordings API

### Endpoint
```
https://atlas.cyclomedia.com/Recordings/wfs
```
(Also seen as `https://atlasapi.cyclomedia.com/api/recording/wfs`)

### Authentication
- HTTP Basic Auth (username + password)
- API Key (passed as parameter)
- OAuth (optional)

### Feature Type
```
atlas:Recording
```

### Supported Operations
| Operation | Description |
|-----------|-------------|
| `GetCapabilities` | List available services |
| `DescribeFeatureType` | Schema for `atlas:Recording` |
| `GetFeature` | Query recordings with OGC filters |

### Output Formats
- `text/xml; subtype=gml/3.1.1` (default)
- `application/json` (GeoJSON)

---

## Recording Feature Properties

From the WFS `DescribeFeatureType` and observed API responses:

| Property | Type | Description |
|----------|------|-------------|
| `imageId` | string | Unique recording identifier (e.g. "W0DEKOEL") |
| `recordedAt` | dateTime | ISO 8601 capture timestamp with timezone offset |
| `location` | gml:Point | Geographic coordinates (lon, lat, height) in requested SRS |
| `height` | double | Camera height above ground (meters) |
| `yaw` | double | Rotation around vertical axis (degrees) |
| `pitch` | double | Rotation around horizontal axis (degrees) |
| `roll` | double | Rotation around image axis (degrees) |
| `expiredAt` | dateTime | Expiry timestamp (null = active) |

### Extended Properties (from Cyclomedia catalog CSVs)

These are available in the pulled recording catalogs at `/share/ju/cyclomedia/pull/`:

| Property | Type | Description |
|----------|------|-------------|
| `recordedAt` | ISO 8601 | e.g. `2025-05-08T11:45:14.0700000-04:00` |
| `recorderDirection` | double | Vehicle heading in degrees (direction of travel) |
| `orientation` | double | Camera yaw in radians |
| `yawDegrees` | double | Camera yaw in degrees |
| `orientationPrecision` | double | Yaw uncertainty (radians) |
| `yawPrecisionDegrees` | double | Yaw uncertainty (degrees) |
| `statePlaneX` | double | State plane X coordinate |
| `statePlaneY` | double | State plane Y coordinate |
| `locationSRS` | string | Coordinate reference system (e.g. `urn:x-ogc:def:crs:EPSG:3857`) |
| `latitudePrecision` | double | Lat precision in meters |
| `longitudePrecision` | double | Lon precision in meters |
| `heightPrecision` | double | Height precision in meters |
| `productType` | string | e.g. "Cyclorama" |
| `hasDepthMap` | boolean | Depth map availability |
| `tileSchema` | string | e.g. "Dcr9Tiling" |
| `year` | int | Capture year |

---

## WFS Query Examples

### GetFeature by Bounding Box (POST, XML)
```xml
<wfs:GetFeature service="WFS" version="1.1.0"
    resultType="results"
    outputFormat="text/xml; subtype=gml/3.1.1"
    xmlns:wfs="http://www.opengis.net/wfs">
  <wfs:Query typeName="atlas:Recording" srsName="EPSG:28992"
      xmlns:atlas="http://www.cyclomedia.com/atlas">
    <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
      <ogc:And>
        <ogc:BBOX>
          <gml:Envelope srsName="EPSG:28992"
              xmlns:gml="http://www.opengis.net/gml">
            <gml:lowerCorner>122882.49 452226.43</gml:lowerCorner>
            <gml:upperCorner>123420.42 452363.27</gml:upperCorner>
          </gml:Envelope>
        </ogc:BBOX>
        <ogc:PropertyIsNull>
          <ogc:PropertyName>expiredAt</ogc:PropertyName>
        </ogc:PropertyIsNull>
      </ogc:And>
    </ogc:Filter>
  </wfs:Query>
</wfs:GetFeature>
```

### GetFeature by Bounding Box (GET, GeoJSON)
```
https://atlas.cyclomedia.com/Recordings/wfs?service=WFS&version=1.1.0&request=GetFeature&typename=atlas:Recording&srsname=EPSG:26918&outputformat=application/json&BBOX=583000,4507000,584000,4508000,EPSG:26918
```

### DWithin (Proximity) Filter
```xml
<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
  <ogc:DWithin>
    <ogc:PropertyName>location</ogc:PropertyName>
    <gml:Point srsName="EPSG:26918" xmlns:gml="http://www.opengis.net/gml">
      <gml:pos>583500 4507500</gml:pos>
    </gml:Point>
    <ogc:Distance units="m">50</ogc:Distance>
  </ogc:DWithin>
</ogc:Filter>
```

### Supported OGC Filter Operators
- **Spatial:** BBOX, DWithin, Contains, Intersects
- **Comparison:** EqualTo, NotEqualTo, LessThan, GreaterThan, PropertyIsNull
- **Logical:** And, Or, Not

---

## Street Smart JavaScript API

### NPM Package
```
npm install @cyclomedia/streetsmart-api
```
(Current version: 26.1.1)

### CDN
```
https://streetsmart.cyclomedia.com/api/v26.1/StreetSmartApi.js
```

### Initialization (from geocoder.nyc/streetview)
```javascript
StreetSmartApi.init({
  targetElement: document.getElementById('streetsmartApi'),
  username, password, apiKey,
  srs: 'EPSG:26918',        // NAD83 UTM Zone 18N (NYC)
  locale: 'en-us',
  configurationUrl: 'https://atlas.cyclomedia.com/configuration',
  addressSettings: { locale: "en", database: "Nokia" }
});
```

### Opening a Panorama
```javascript
StreetSmartApi.open(coordinate_string, {
  viewerType: StreetSmartApi.ViewerType.PANORAMA_ZOOM,
  srs: 'EPSG:26918',
  panoramaViewer: {
    closable: false,
    maximizable: true,
    replace: true,
    recordingsVisible: true,
    navbarVisible: true,
    timeTravelVisible: true,
  }
}).then(result => {
  window.panoramaViewer = result[0];
  // Listen for navigation events (recording changes)
  window.panoramaViewer.on(
    StreetSmartApi.Events.panoramaViewer.VIEW_CHANGE,
    changeview
  );
});
```

### Key Viewer Methods
```javascript
// Get current recording (returns {imageId, xyz, srs, ...})
panoramaViewer.getRecording()

// Get current orientation
panoramaViewer.getOrientation()  // → {yaw, pitch, hFov}

// Open specific recording
StreetSmartApi.open(imageId, options)

// Events
StreetSmartApi.Events.panoramaViewer.VIEW_CHANGE      // orientation changed
StreetSmartApi.Events.panoramaViewer.RECORDING_CLICK   // clicked a recording dot
StreetSmartApi.Events.panoramaViewer.RECORDING_LOADED  // new recording loaded
StreetSmartApi.Events.panoramaViewer.IMAGE_CHANGE      // navigated to new image
```

### WFS Client Usage (from geocoder.nyc)
```javascript
// Instantiate WFS client
var wfsClient = new WFSClient(
  "https://atlas.cyclomedia.com/Recordings/wfs",
  "atlas:Recording",
  "EPSG:26918",
  "",       // filter (empty = none)
  apiKey
);

// Query recordings in map viewport
var extent = map.getView().calculateExtent(map.getSize());
wfsClient.loadBbox(extent[0], extent[1], extent[2], extent[3], callback, username, password);

// Callback receives wfsClient.recordingList → [{lon, lat, ...}, ...]
```

### Viewing Cone Geometry (geocoder.nyc)
```javascript
// The viewer tracks orientation as a cone on the map
function changeview() {
  const rl = window.panoramaViewer.getRecording();  // {xyz, srs}
  const orientation = window.panoramaViewer.getOrientation();
  // orientation.yaw = viewing direction in degrees
  // orientation.hFov = horizontal field of view in degrees
  // rl.xyz = [x, y, z] in SRS coordinates
}
```

---

## The Actual Navigation Algorithm (Extracted from StreetSmartApi.js v26.1)

The Street Smart viewer has **two parallel navigation systems**, both reverse-engineered
from the minified JS bundle at `streetsmart.cyclomedia.com/api/v26.1/StreetSmartApi.js`:

### System 1: Arrow-Key Navigation (Spatial Scoring)

On every `VIEW_CHANGE` event, `_updateRecordingScores()` runs:

```javascript
// Extracted from StreetSmartApi.js (deobfuscated)
_updateRecordingScores() {
  const camera = this._viewer._camera;
  const recordings = this._viewedRecordings;  // loaded via requestWithinRadius()
  const yaw = camera.yaw;  // current viewer heading (radians)

  let bestForwardScore = -1, bestForwardIdx = -1;
  let bestBackwardScore = -1, bestBackwardIdx = -1;

  for (let i = 0; i < recordings.length; i++) {
    const rec = recordings[i];
    const relYaw = rec.relativeYaw;        // bearing from active to candidate (radians)
    const relDist = rec.relativeDistance;    // meters

    // Forward score: how well does this candidate match "ahead"?
    const angleFwd = Math.acos(Math.cos(yaw - relYaw));
    const scoreFwd = S(relDist, angleFwd);
    rec.scoreForward = scoreFwd;
    rec.forwardTarget = false;
    if (scoreFwd > 0 && scoreFwd > bestForwardScore) {
      bestForwardScore = scoreFwd;
      bestForwardIdx = i;
    }

    // Backward score: how well does this candidate match "behind"?
    const angleBwd = Math.acos(Math.cos(yaw - relYaw + Math.PI));
    const scoreBwd = S(relDist, angleBwd);
    rec.scoreBackward = scoreBwd;
    rec.backwardTarget = false;
    if (scoreBwd > 0 && scoreBwd > bestBackwardScore) {
      bestBackwardScore = scoreBwd;
      bestBackwardIdx = i;
    }
  }

  // Set the winners as forward/backward targets
  if (bestBackwardIdx >= 0) recordings[bestBackwardIdx].backwardTarget = true;
  if (bestForwardIdx >= 0)  recordings[bestForwardIdx].forwardTarget = true;

  this._forwardTarget = bestForwardIdx >= 0 ? recordings[bestForwardIdx] : null;
  this._backwardTarget = bestBackwardIdx >= 0 ? recordings[bestBackwardIdx] : null;
}
```

**The scoring function `S(distance, angle)`:**

```javascript
// THE key formula — controls all spatial navigation
S = (distance, angle) => angle > Math.PI/4 ? -1 : (1 - angle/Math.PI) / (1 + 0.1 * Math.sqrt(distance))
```

- `distance`: meters from active recording to candidate
- `angle`: radians, angular deviation from desired direction
- Returns **-1** if angle > 45° (reject), otherwise a score in (0, 1]
- Penalizes both distance (√dist) and angular deviation (linear in angle)
- At 0° deviation, 0m distance: score = 1.0 (perfect)
- At 0° deviation, 100m distance: score = 0.5
- At 45° deviation, any distance: score = -1 (rejected)

**How recordings are loaded:**

```javascript
// Aperture loads nearby recordings via WFS within a configurable radius
_triggerRecordingLoad(client) {
  const {xyz, srs} = this._activeRecording;
  client.requestWithinRadius(xyz[0], xyz[1], this._recordingRequestRadius, srs, options);
}
```

**Candidate filtering:**

```javascript
_initializeViewedRecordings(recordings) {
  const active = this._activeRecording;
  const drawRadiusSq = this._recordingDrawRadius ** 2;
  const candidates = [];

  for (const rec of recordings) {
    if (rec.id === active.id) continue;
    const relXyz = /* project to ENU coords */;
    let distSq = sqrLen(relXyz);
    if (distSq <= drawRadiusSq && this._checkCandidateAgainstViewedRecordingOptions(rec)) {
      rec.relativeDistance = Math.sqrt(distSq);
      rec.relativeYaw = Math.acos(relXyz[1] / rec.relativeDistance);  // bearing
      if (relXyz[0] < 0) rec.relativeYaw = 2*PI - rec.relativeYaw;
      candidates.push(rec);
    }
  }
  return candidates;
}

// Date range filtering (time travel feature)
_checkCandidateAgainstViewedRecordingOptions(rec) {
  const {dateRange} = this._viewedRecordingOptions || {};
  return !dateRange || (dateRange.from <= rec.recordedAt && rec.recordedAt <= dateRange.to);
}
```

**Keyboard handler:**

```javascript
// ArrowUp/W = forward, ArrowDown/S = backward
case "ArrowUp": case "W":
  target = this._forwardTarget;
  if (target) this.openPanoramaFromRecording(target);
  break;
case "ArrowDown": case "S":
  target = this._backwardTarget;
  if (target) this.openPanoramaFromRecording(target);
  break;
```

### System 2: Cruise/Route Mode (LRS Linear Referencing)

The "play" button uses pre-computed routes from a **Linear Referencing System (LRS)** WFS:

```javascript
// Build LRS WFS query URL
_buildUrl() {
  const {onlineResource, typeName} = this.props.lrsService;
  return `${onlineResource}?service=WFS&version=1.1.0&typename=${typeName}`
    + `&outputFormat=application/json&REQUEST=GetFeature`;
}

// Query route for current recording
_getRecordingRouteRecording(recordingId) {
  return `${this._buildUrl()}&cql_filter=record='${recordingId}'&maxFeatures=1`;
}

// Get full route by routeid + direction + published_at
_getRecordingRoute(routeid, direction, published_at) {
  return `${this._buildUrl()}&cql_filter=routeid='${routeid}'`
    + ` AND direction='${direction}' AND published_at='${published_at}'`
    + ` AND frame is not null&sortBy=frame`;
}

// Navigation is simply: routeList[currentIndex ± step]
_getRecordingInfo(routeList, currentRecordId) {
  const idx = routeList.findIndex(r => r.properties.record === currentRecordId);
  const step = this.state.step;
  return {
    previousRecordingId: idx > 0 ? routeList[idx - step]?.record : undefined,
    nextRecordingId: idx < routeList.length-1 ? routeList[idx + step]?.record : undefined,
  };
}
```

**Route properties:** `routeid`, `direction`, `frame` (sequence number), `record` (imageId),
`published_at`, `mileage`, `routename`

### Recording Object Properties

```javascript
// From the Recording constructor
{
  id: string,                    // required
  xyz: [x, y, z],              // required, projected coordinates
  srs: string,                  // required, e.g. "EPSG:26918"
  groundLevelOffset: 0,
  recorderDirection: 0,          // vehicle heading (degrees)
  orientation: 0,                // camera yaw (radians)
  recordedAt: Date | null,
  expiredAt: Date | null,
  year: 0,
  productType: string,
  orientationPitch: float | null,
  orientationRoll: float | null,
  orientationYaw: float | null,
  hasDepthMap: boolean,
  // Computed at runtime:
  relativeXyz: [x, y, z],       // ENU offset from active recording
  relativeYaw: float,            // bearing in radians
  relativeDistance: float,        // meters
  scoreForward: float,
  scoreBackward: float,
  forwardTarget: boolean,
  backwardTarget: boolean,
}
```

### Our Implementation

The `build_trajectory_graph()` function combines both systems:

1. **Phase 1 (LRS analog):** Sort by `recordedAt`, segment into passes using
   time/distance/heading gap thresholds, chain sequentially within each pass.
2. **Phase 2 (Spatial scoring):** For each recording, find cross-pass neighbors
   within `spatial_radius_m` and score them with the **actual Street Smart `S()` function**.
   Keep top `max_spatial_neighbors` per node.
3. **Phase 3 (Component bridging):** Bridge any disconnected components to ensure
   global traversability.

### Key Parameters (from Cyclomedia capture patterns)
- **Capture interval:** ~5 meters along roads
- **Time between captures:** ~0.5-1 seconds at driving speed (median 0.66s)
- **Pass gap threshold:** 10s covers 99%+ of within-pass gaps
- **Spatial scoring radius:** 30m (covers adjacent streets)

---

## Data Sources for Implementation

### Already Available
- Recording catalogs with timestamps: `/share/ju/cyclomedia/pull/recordings_*_2025_part*.csv`
  - Contains: `imageId, lon, lat, recordedAt, recorderDirection, yawDegrees, orientation`
- Current parquet data: paths in `dagspaces/urbanroamvqa/conf/data/cyclomedia_manhattan_2025.yaml`

### Need to Enrich
The current parquet files lack `recordedAt` and `recorderDirection`. These must be joined from the catalog CSVs using `imageId`/`recording_id` as the join key.

---

## External References

| Resource | URL |
|----------|-----|
| Developer Portal | https://developer.cyclomedia.com/our-apis/street-smart/ |
| JS API Docs | https://developer.cyclomedia.com/documentation/street-smart/js-api/ |
| NPM Package | https://www.npmjs.com/package/@cyclomedia/streetsmart-api |
| WFS Docs (PDF) | https://docs.cyclomedia.com/Atlas/2020/Atlas%20WFS%20Recordings%20Service.pdf |
| API Overview (PDF) | https://docs.cyclomedia.com/Atlas/2020/An%20introduction%20to%20the%20Cyclomedia%20APIs.pdf |
| Dev Guidelines (PDF) | https://docs.cyclomedia.com/Atlas/2020/The%20Street%20Smart%20API%20%20Guideline%20for%20developers.pdf |
| NYC Integration | https://www.geocoder.nyc/streetview |
| GitHub: ArcGIS Widget | https://github.com/cyclomedia/streetsmart-aol-widget |
| GitHub: .NET Wrapper | https://github.com/cyclomedia/streetsmart-dotnet |
| GitHub: NYC Planning | https://github.com/NYCPlanning/labs-cyclomedia-service |
| GitHub: Philly Mapboard | https://github.com/CityOfPhiladelphia/cyclomedia-mapboard |
