import ProjectEulerSolutions.Lib.PrimePairLarge.Graph002
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row523_eq_data : row523 = Data.row523 := rfl
def node523 : Graph := .node 523 row523 node233 node857
set_option maxRecDepth 4096 in
theorem node523_covers : Budget.Covers PrimePairVertices.largeTree 26034 node523 :=
  Budget.node_covers _ _ 523 row523 node233 node857
    node233_covers node857_covers (by decide) (by decide) row523_covers
theorem node523_eq_data : node523 = Data.node523 := by
  have branches : (node233, node857) = (Data.node233, Data.node857) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node233_eq_data node857_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 523 row children.1 children.2) row523_eq_data branches
set_option maxRecDepth 4096 in
theorem row1201_eq_data : row1201 = Data.row1201 := rfl
def node1201 : Graph := .node 1201 row1201 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1201_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1201 :=
  Budget.node_covers _ _ 1201 row1201 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1201_covers
theorem node1201_eq_data : node1201 = Data.node1201 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1201 row children.1 children.2) row1201_eq_data branches
set_option maxRecDepth 4096 in
theorem row1217_eq_data : row1217 = Data.row1217 := rfl
def node1217 : Graph := .node 1217 row1217 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1217_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1217 :=
  Budget.node_covers _ _ 1217 row1217 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1217_covers
theorem node1217_eq_data : node1217 = Data.node1217 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1217 row children.1 children.2) row1217_eq_data branches
set_option maxRecDepth 4096 in
theorem row1213_eq_data : row1213 = Data.row1213 := rfl
def node1213 : Graph := .node 1213 row1213 node1201 node1217
set_option maxRecDepth 4096 in
theorem node1213_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1213 :=
  Budget.node_covers _ _ 1213 row1213 node1201 node1217
    node1201_covers node1217_covers (by decide) (by decide) row1213_covers
theorem node1213_eq_data : node1213 = Data.node1213 := by
  have branches : (node1201, node1217) = (Data.node1201, Data.node1217) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1201_eq_data node1217_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1213 row children.1 children.2) row1213_eq_data branches
set_option maxRecDepth 4096 in
theorem row1229_eq_data : row1229 = Data.row1229 := rfl
def node1229 : Graph := .node 1229 row1229 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1229_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1229 :=
  Budget.node_covers _ _ 1229 row1229 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1229_covers
theorem node1229_eq_data : node1229 = Data.node1229 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1229 row children.1 children.2) row1229_eq_data branches
set_option maxRecDepth 4096 in
theorem row1231_eq_data : row1231 = Data.row1231 := rfl
def node1231 : Graph := .node 1231 row1231 node1229 Graph.empty
set_option maxRecDepth 4096 in
theorem node1231_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1231 :=
  Budget.node_covers _ _ 1231 row1231 node1229 Graph.empty
    node1229_covers empty_covers (by decide) (by decide) row1231_covers
theorem node1231_eq_data : node1231 = Data.node1231 := by
  have branches : (node1229, Graph.empty) = (Data.node1229, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1229_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1231 row children.1 children.2) row1231_eq_data branches
set_option maxRecDepth 4096 in
theorem row1223_eq_data : row1223 = Data.row1223 := rfl
def node1223 : Graph := .node 1223 row1223 node1213 node1231
set_option maxRecDepth 4096 in
theorem node1223_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1223 :=
  Budget.node_covers _ _ 1223 row1223 node1213 node1231
    node1213_covers node1231_covers (by decide) (by decide) row1223_covers
theorem node1223_eq_data : node1223 = Data.node1223 := by
  have branches : (node1213, node1231) = (Data.node1213, Data.node1231) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1213_eq_data node1231_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1223 row children.1 children.2) row1223_eq_data branches
set_option maxRecDepth 4096 in
theorem row1249_eq_data : row1249 = Data.row1249 := rfl
def node1249 : Graph := .node 1249 row1249 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1249_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1249 :=
  Budget.node_covers _ _ 1249 row1249 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1249_covers
theorem node1249_eq_data : node1249 = Data.node1249 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1249 row children.1 children.2) row1249_eq_data branches
set_option maxRecDepth 4096 in
theorem row1259_eq_data : row1259 = Data.row1259 := rfl
def node1259 : Graph := .node 1259 row1259 node1249 Graph.empty
set_option maxRecDepth 4096 in
theorem node1259_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1259 :=
  Budget.node_covers _ _ 1259 row1259 node1249 Graph.empty
    node1249_covers empty_covers (by decide) (by decide) row1259_covers
theorem node1259_eq_data : node1259 = Data.node1259 := by
  have branches : (node1249, Graph.empty) = (Data.node1249, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1249_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1259 row children.1 children.2) row1259_eq_data branches
set_option maxRecDepth 4096 in
theorem row1279_eq_data : row1279 = Data.row1279 := rfl
def node1279 : Graph := .node 1279 row1279 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1279_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1279 :=
  Budget.node_covers _ _ 1279 row1279 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1279_covers
theorem node1279_eq_data : node1279 = Data.node1279 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1279 row children.1 children.2) row1279_eq_data branches
set_option maxRecDepth 4096 in
theorem row1283_eq_data : row1283 = Data.row1283 := rfl
def node1283 : Graph := .node 1283 row1283 node1279 Graph.empty
set_option maxRecDepth 4096 in
theorem node1283_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1283 :=
  Budget.node_covers _ _ 1283 row1283 node1279 Graph.empty
    node1279_covers empty_covers (by decide) (by decide) row1283_covers
theorem node1283_eq_data : node1283 = Data.node1283 := by
  have branches : (node1279, Graph.empty) = (Data.node1279, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1279_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1283 row children.1 children.2) row1283_eq_data branches
set_option maxRecDepth 4096 in
theorem row1277_eq_data : row1277 = Data.row1277 := rfl
def node1277 : Graph := .node 1277 row1277 node1259 node1283
set_option maxRecDepth 4096 in
theorem node1277_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1277 :=
  Budget.node_covers _ _ 1277 row1277 node1259 node1283
    node1259_covers node1283_covers (by decide) (by decide) row1277_covers
theorem node1277_eq_data : node1277 = Data.node1277 := by
  have branches : (node1259, node1283) = (Data.node1259, Data.node1283) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1259_eq_data node1283_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1277 row children.1 children.2) row1277_eq_data branches
set_option maxRecDepth 4096 in
theorem row1237_eq_data : row1237 = Data.row1237 := rfl
def node1237 : Graph := .node 1237 row1237 node1223 node1277
set_option maxRecDepth 4096 in
theorem node1237_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1237 :=
  Budget.node_covers _ _ 1237 row1237 node1223 node1277
    node1223_covers node1277_covers (by decide) (by decide) row1237_covers
theorem node1237_eq_data : node1237 = Data.node1237 := by
  have branches : (node1223, node1277) = (Data.node1223, Data.node1277) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1223_eq_data node1277_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1237 row children.1 children.2) row1237_eq_data branches
set_option maxRecDepth 4096 in
theorem row1291_eq_data : row1291 = Data.row1291 := rfl
def node1291 : Graph := .node 1291 row1291 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1291_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1291 :=
  Budget.node_covers _ _ 1291 row1291 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1291_covers
theorem node1291_eq_data : node1291 = Data.node1291 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1291 row children.1 children.2) row1291_eq_data branches
set_option maxRecDepth 4096 in
theorem row1297_eq_data : row1297 = Data.row1297 := rfl
def node1297 : Graph := .node 1297 row1297 node1291 Graph.empty
set_option maxRecDepth 4096 in
theorem node1297_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1297 :=
  Budget.node_covers _ _ 1297 row1297 node1291 Graph.empty
    node1291_covers empty_covers (by decide) (by decide) row1297_covers
theorem node1297_eq_data : node1297 = Data.node1297 := by
  have branches : (node1291, Graph.empty) = (Data.node1291, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1291_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1297 row children.1 children.2) row1297_eq_data branches
set_option maxRecDepth 4096 in
theorem row1303_eq_data : row1303 = Data.row1303 := rfl
def node1303 : Graph := .node 1303 row1303 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1303_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1303 :=
  Budget.node_covers _ _ 1303 row1303 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1303_covers
theorem node1303_eq_data : node1303 = Data.node1303 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1303 row children.1 children.2) row1303_eq_data branches
set_option maxRecDepth 4096 in
theorem row1307_eq_data : row1307 = Data.row1307 := rfl
def node1307 : Graph := .node 1307 row1307 node1303 Graph.empty
set_option maxRecDepth 4096 in
theorem node1307_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1307 :=
  Budget.node_covers _ _ 1307 row1307 node1303 Graph.empty
    node1303_covers empty_covers (by decide) (by decide) row1307_covers
theorem node1307_eq_data : node1307 = Data.node1307 := by
  have branches : (node1303, Graph.empty) = (Data.node1303, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1303_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1307 row children.1 children.2) row1307_eq_data branches
set_option maxRecDepth 4096 in
theorem row1301_eq_data : row1301 = Data.row1301 := rfl
def node1301 : Graph := .node 1301 row1301 node1297 node1307
set_option maxRecDepth 4096 in
theorem node1301_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1301 :=
  Budget.node_covers _ _ 1301 row1301 node1297 node1307
    node1297_covers node1307_covers (by decide) (by decide) row1301_covers
theorem node1301_eq_data : node1301 = Data.node1301 := by
  have branches : (node1297, node1307) = (Data.node1297, Data.node1307) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1297_eq_data node1307_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1301 row children.1 children.2) row1301_eq_data branches
set_option maxRecDepth 4096 in
theorem row1321_eq_data : row1321 = Data.row1321 := rfl
def node1321 : Graph := .node 1321 row1321 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1321_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1321 :=
  Budget.node_covers _ _ 1321 row1321 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1321_covers
theorem node1321_eq_data : node1321 = Data.node1321 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1321 row children.1 children.2) row1321_eq_data branches
set_option maxRecDepth 4096 in
theorem row1327_eq_data : row1327 = Data.row1327 := rfl
def node1327 : Graph := .node 1327 row1327 node1321 Graph.empty
set_option maxRecDepth 4096 in
theorem node1327_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1327 :=
  Budget.node_covers _ _ 1327 row1327 node1321 Graph.empty
    node1321_covers empty_covers (by decide) (by decide) row1327_covers
theorem node1327_eq_data : node1327 = Data.node1327 := by
  have branches : (node1321, Graph.empty) = (Data.node1321, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1321_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1327 row children.1 children.2) row1327_eq_data branches
set_option maxRecDepth 4096 in
theorem row1367_eq_data : row1367 = Data.row1367 := rfl
def node1367 : Graph := .node 1367 row1367 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1367_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1367 :=
  Budget.node_covers _ _ 1367 row1367 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1367_covers
theorem node1367_eq_data : node1367 = Data.node1367 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1367 row children.1 children.2) row1367_eq_data branches
set_option maxRecDepth 4096 in
theorem row1373_eq_data : row1373 = Data.row1373 := rfl
def node1373 : Graph := .node 1373 row1373 node1367 Graph.empty
set_option maxRecDepth 4096 in
theorem node1373_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1373 :=
  Budget.node_covers _ _ 1373 row1373 node1367 Graph.empty
    node1367_covers empty_covers (by decide) (by decide) row1373_covers
theorem node1373_eq_data : node1373 = Data.node1373 := by
  have branches : (node1367, Graph.empty) = (Data.node1367, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1367_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1373 row children.1 children.2) row1373_eq_data branches
set_option maxRecDepth 4096 in
theorem row1361_eq_data : row1361 = Data.row1361 := rfl
def node1361 : Graph := .node 1361 row1361 node1327 node1373
set_option maxRecDepth 4096 in
theorem node1361_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1361 :=
  Budget.node_covers _ _ 1361 row1361 node1327 node1373
    node1327_covers node1373_covers (by decide) (by decide) row1361_covers
theorem node1361_eq_data : node1361 = Data.node1361 := by
  have branches : (node1327, node1373) = (Data.node1327, Data.node1373) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1327_eq_data node1373_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1361 row children.1 children.2) row1361_eq_data branches
set_option maxRecDepth 4096 in
theorem row1319_eq_data : row1319 = Data.row1319 := rfl
def node1319 : Graph := .node 1319 row1319 node1301 node1361
set_option maxRecDepth 4096 in
theorem node1319_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1319 :=
  Budget.node_covers _ _ 1319 row1319 node1301 node1361
    node1301_covers node1361_covers (by decide) (by decide) row1319_covers
theorem node1319_eq_data : node1319 = Data.node1319 := by
  have branches : (node1301, node1361) = (Data.node1301, Data.node1361) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1301_eq_data node1361_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1319 row children.1 children.2) row1319_eq_data branches
set_option maxRecDepth 4096 in
theorem row1289_eq_data : row1289 = Data.row1289 := rfl
def node1289 : Graph := .node 1289 row1289 node1237 node1319
set_option maxRecDepth 4096 in
theorem node1289_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1289 :=
  Budget.node_covers _ _ 1289 row1289 node1237 node1319
    node1237_covers node1319_covers (by decide) (by decide) row1289_covers
theorem node1289_eq_data : node1289 = Data.node1289 := by
  have branches : (node1237, node1319) = (Data.node1237, Data.node1319) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1237_eq_data node1319_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1289 row children.1 children.2) row1289_eq_data branches
set_option maxRecDepth 4096 in
theorem row1399_eq_data : row1399 = Data.row1399 := rfl
def node1399 : Graph := .node 1399 row1399 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1399_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1399 :=
  Budget.node_covers _ _ 1399 row1399 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1399_covers
theorem node1399_eq_data : node1399 = Data.node1399 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1399 row children.1 children.2) row1399_eq_data branches
set_option maxRecDepth 4096 in
theorem row1409_eq_data : row1409 = Data.row1409 := rfl
def node1409 : Graph := .node 1409 row1409 node1399 Graph.empty
set_option maxRecDepth 4096 in
theorem node1409_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1409 :=
  Budget.node_covers _ _ 1409 row1409 node1399 Graph.empty
    node1399_covers empty_covers (by decide) (by decide) row1409_covers
theorem node1409_eq_data : node1409 = Data.node1409 := by
  have branches : (node1399, Graph.empty) = (Data.node1399, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1399_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1409 row children.1 children.2) row1409_eq_data branches
set_option maxRecDepth 4096 in
theorem row1427_eq_data : row1427 = Data.row1427 := rfl
def node1427 : Graph := .node 1427 row1427 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1427_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1427 :=
  Budget.node_covers _ _ 1427 row1427 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1427_covers
theorem node1427_eq_data : node1427 = Data.node1427 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1427 row children.1 children.2) row1427_eq_data branches
set_option maxRecDepth 4096 in
theorem row1429_eq_data : row1429 = Data.row1429 := rfl
def node1429 : Graph := .node 1429 row1429 node1427 Graph.empty
set_option maxRecDepth 4096 in
theorem node1429_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1429 :=
  Budget.node_covers _ _ 1429 row1429 node1427 Graph.empty
    node1427_covers empty_covers (by decide) (by decide) row1429_covers
theorem node1429_eq_data : node1429 = Data.node1429 := by
  have branches : (node1427, Graph.empty) = (Data.node1427, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1427_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1429 row children.1 children.2) row1429_eq_data branches
set_option maxRecDepth 4096 in
theorem row1423_eq_data : row1423 = Data.row1423 := rfl
def node1423 : Graph := .node 1423 row1423 node1409 node1429
set_option maxRecDepth 4096 in
theorem node1423_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1423 :=
  Budget.node_covers _ _ 1423 row1423 node1409 node1429
    node1409_covers node1429_covers (by decide) (by decide) row1423_covers
theorem node1423_eq_data : node1423 = Data.node1423 := by
  have branches : (node1409, node1429) = (Data.node1409, Data.node1429) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1409_eq_data node1429_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1423 row children.1 children.2) row1423_eq_data branches
set_option maxRecDepth 4096 in
theorem row1439_eq_data : row1439 = Data.row1439 := rfl
def node1439 : Graph := .node 1439 row1439 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1439_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1439 :=
  Budget.node_covers _ _ 1439 row1439 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1439_covers
theorem node1439_eq_data : node1439 = Data.node1439 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1439 row children.1 children.2) row1439_eq_data branches
set_option maxRecDepth 4096 in
theorem row1447_eq_data : row1447 = Data.row1447 := rfl
def node1447 : Graph := .node 1447 row1447 node1439 Graph.empty
set_option maxRecDepth 4096 in
theorem node1447_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1447 :=
  Budget.node_covers _ _ 1447 row1447 node1439 Graph.empty
    node1439_covers empty_covers (by decide) (by decide) row1447_covers
theorem node1447_eq_data : node1447 = Data.node1447 := by
  have branches : (node1439, Graph.empty) = (Data.node1439, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1439_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1447 row children.1 children.2) row1447_eq_data branches
set_option maxRecDepth 4096 in
theorem row1453_eq_data : row1453 = Data.row1453 := rfl
def node1453 : Graph := .node 1453 row1453 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1453_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1453 :=
  Budget.node_covers _ _ 1453 row1453 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1453_covers
theorem node1453_eq_data : node1453 = Data.node1453 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1453 row children.1 children.2) row1453_eq_data branches
set_option maxRecDepth 4096 in
theorem row1459_eq_data : row1459 = Data.row1459 := rfl
def node1459 : Graph := .node 1459 row1459 node1453 Graph.empty
set_option maxRecDepth 4096 in
theorem node1459_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1459 :=
  Budget.node_covers _ _ 1459 row1459 node1453 Graph.empty
    node1453_covers empty_covers (by decide) (by decide) row1459_covers
theorem node1459_eq_data : node1459 = Data.node1459 := by
  have branches : (node1453, Graph.empty) = (Data.node1453, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1453_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1459 row children.1 children.2) row1459_eq_data branches
set_option maxRecDepth 4096 in
theorem row1451_eq_data : row1451 = Data.row1451 := rfl
def node1451 : Graph := .node 1451 row1451 node1447 node1459
set_option maxRecDepth 4096 in
theorem node1451_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1451 :=
  Budget.node_covers _ _ 1451 row1451 node1447 node1459
    node1447_covers node1459_covers (by decide) (by decide) row1451_covers
theorem node1451_eq_data : node1451 = Data.node1451 := by
  have branches : (node1447, node1459) = (Data.node1447, Data.node1459) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1447_eq_data node1459_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1451 row children.1 children.2) row1451_eq_data branches
set_option maxRecDepth 4096 in
theorem row1433_eq_data : row1433 = Data.row1433 := rfl
def node1433 : Graph := .node 1433 row1433 node1423 node1451
set_option maxRecDepth 4096 in
theorem node1433_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1433 :=
  Budget.node_covers _ _ 1433 row1433 node1423 node1451
    node1423_covers node1451_covers (by decide) (by decide) row1433_covers
theorem node1433_eq_data : node1433 = Data.node1433 := by
  have branches : (node1423, node1451) = (Data.node1423, Data.node1451) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1423_eq_data node1451_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1433 row children.1 children.2) row1433_eq_data branches
set_option maxRecDepth 4096 in
theorem row1481_eq_data : row1481 = Data.row1481 := rfl
def node1481 : Graph := .node 1481 row1481 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1481_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1481 :=
  Budget.node_covers _ _ 1481 row1481 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1481_covers
theorem node1481_eq_data : node1481 = Data.node1481 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1481 row children.1 children.2) row1481_eq_data branches
set_option maxRecDepth 4096 in
theorem row1483_eq_data : row1483 = Data.row1483 := rfl
def node1483 : Graph := .node 1483 row1483 node1481 Graph.empty
set_option maxRecDepth 4096 in
theorem node1483_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1483 :=
  Budget.node_covers _ _ 1483 row1483 node1481 Graph.empty
    node1481_covers empty_covers (by decide) (by decide) row1483_covers
theorem node1483_eq_data : node1483 = Data.node1483 := by
  have branches : (node1481, Graph.empty) = (Data.node1481, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1481_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1483 row children.1 children.2) row1483_eq_data branches
set_option maxRecDepth 4096 in
theorem row1489_eq_data : row1489 = Data.row1489 := rfl
def node1489 : Graph := .node 1489 row1489 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1489_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1489 :=
  Budget.node_covers _ _ 1489 row1489 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1489_covers
theorem node1489_eq_data : node1489 = Data.node1489 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1489 row children.1 children.2) row1489_eq_data branches
set_option maxRecDepth 4096 in
theorem row1493_eq_data : row1493 = Data.row1493 := rfl
def node1493 : Graph := .node 1493 row1493 node1489 Graph.empty
set_option maxRecDepth 4096 in
theorem node1493_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1493 :=
  Budget.node_covers _ _ 1493 row1493 node1489 Graph.empty
    node1489_covers empty_covers (by decide) (by decide) row1493_covers
theorem node1493_eq_data : node1493 = Data.node1493 := by
  have branches : (node1489, Graph.empty) = (Data.node1489, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1489_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1493 row children.1 children.2) row1493_eq_data branches
set_option maxRecDepth 4096 in
theorem row1487_eq_data : row1487 = Data.row1487 := rfl
def node1487 : Graph := .node 1487 row1487 node1483 node1493
set_option maxRecDepth 4096 in
theorem node1487_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1487 :=
  Budget.node_covers _ _ 1487 row1487 node1483 node1493
    node1483_covers node1493_covers (by decide) (by decide) row1487_covers
theorem node1487_eq_data : node1487 = Data.node1487 := by
  have branches : (node1483, node1493) = (Data.node1483, Data.node1493) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1483_eq_data node1493_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1487 row children.1 children.2) row1487_eq_data branches
set_option maxRecDepth 4096 in
theorem row1511_eq_data : row1511 = Data.row1511 := rfl
def node1511 : Graph := .node 1511 row1511 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1511_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1511 :=
  Budget.node_covers _ _ 1511 row1511 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1511_covers
theorem node1511_eq_data : node1511 = Data.node1511 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1511 row children.1 children.2) row1511_eq_data branches
set_option maxRecDepth 4096 in
theorem row1523_eq_data : row1523 = Data.row1523 := rfl
def node1523 : Graph := .node 1523 row1523 node1511 Graph.empty
set_option maxRecDepth 4096 in
theorem node1523_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1523 :=
  Budget.node_covers _ _ 1523 row1523 node1511 Graph.empty
    node1511_covers empty_covers (by decide) (by decide) row1523_covers
theorem node1523_eq_data : node1523 = Data.node1523 := by
  have branches : (node1511, Graph.empty) = (Data.node1511, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1511_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1523 row children.1 children.2) row1523_eq_data branches
set_option maxRecDepth 4096 in
theorem row1543_eq_data : row1543 = Data.row1543 := rfl
def node1543 : Graph := .node 1543 row1543 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1543_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1543 :=
  Budget.node_covers _ _ 1543 row1543 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1543_covers
theorem node1543_eq_data : node1543 = Data.node1543 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1543 row children.1 children.2) row1543_eq_data branches
set_option maxRecDepth 4096 in
theorem row1549_eq_data : row1549 = Data.row1549 := rfl
def node1549 : Graph := .node 1549 row1549 node1543 Graph.empty
set_option maxRecDepth 4096 in
theorem node1549_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1549 :=
  Budget.node_covers _ _ 1549 row1549 node1543 Graph.empty
    node1543_covers empty_covers (by decide) (by decide) row1549_covers
theorem node1549_eq_data : node1549 = Data.node1549 := by
  have branches : (node1543, Graph.empty) = (Data.node1543, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1543_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1549 row children.1 children.2) row1549_eq_data branches
set_option maxRecDepth 4096 in
theorem row1531_eq_data : row1531 = Data.row1531 := rfl
def node1531 : Graph := .node 1531 row1531 node1523 node1549
set_option maxRecDepth 4096 in
theorem node1531_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1531 :=
  Budget.node_covers _ _ 1531 row1531 node1523 node1549
    node1523_covers node1549_covers (by decide) (by decide) row1531_covers
theorem node1531_eq_data : node1531 = Data.node1531 := by
  have branches : (node1523, node1549) = (Data.node1523, Data.node1549) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1523_eq_data node1549_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1531 row children.1 children.2) row1531_eq_data branches
set_option maxRecDepth 4096 in
theorem row1499_eq_data : row1499 = Data.row1499 := rfl
def node1499 : Graph := .node 1499 row1499 node1487 node1531
set_option maxRecDepth 4096 in
theorem node1499_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1499 :=
  Budget.node_covers _ _ 1499 row1499 node1487 node1531
    node1487_covers node1531_covers (by decide) (by decide) row1499_covers
theorem node1499_eq_data : node1499 = Data.node1499 := by
  have branches : (node1487, node1531) = (Data.node1487, Data.node1531) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1487_eq_data node1531_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1499 row children.1 children.2) row1499_eq_data branches
set_option maxRecDepth 4096 in
theorem row1471_eq_data : row1471 = Data.row1471 := rfl
def node1471 : Graph := .node 1471 row1471 node1433 node1499
set_option maxRecDepth 4096 in
theorem node1471_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1471 :=
  Budget.node_covers _ _ 1471 row1471 node1433 node1499
    node1433_covers node1499_covers (by decide) (by decide) row1471_covers
theorem node1471_eq_data : node1471 = Data.node1471 := by
  have branches : (node1433, node1499) = (Data.node1433, Data.node1499) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1433_eq_data node1499_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1471 row children.1 children.2) row1471_eq_data branches
set_option maxRecDepth 4096 in
theorem row1381_eq_data : row1381 = Data.row1381 := rfl
def node1381 : Graph := .node 1381 row1381 node1289 node1471
set_option maxRecDepth 4096 in
theorem node1381_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1381 :=
  Budget.node_covers _ _ 1381 row1381 node1289 node1471
    node1289_covers node1471_covers (by decide) (by decide) row1381_covers
theorem node1381_eq_data : node1381 = Data.node1381 := by
  have branches : (node1289, node1471) = (Data.node1289, Data.node1471) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1289_eq_data node1471_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1381 row children.1 children.2) row1381_eq_data branches
set_option maxRecDepth 4096 in
theorem row1559_eq_data : row1559 = Data.row1559 := rfl
def node1559 : Graph := .node 1559 row1559 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1559_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1559 :=
  Budget.node_covers _ _ 1559 row1559 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1559_covers
theorem node1559_eq_data : node1559 = Data.node1559 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1559 row children.1 children.2) row1559_eq_data branches
set_option maxRecDepth 4096 in
theorem row1567_eq_data : row1567 = Data.row1567 := rfl
def node1567 : Graph := .node 1567 row1567 node1559 Graph.empty
set_option maxRecDepth 4096 in
theorem node1567_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1567 :=
  Budget.node_covers _ _ 1567 row1567 node1559 Graph.empty
    node1559_covers empty_covers (by decide) (by decide) row1567_covers
theorem node1567_eq_data : node1567 = Data.node1567 := by
  have branches : (node1559, Graph.empty) = (Data.node1559, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1559_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1567 row children.1 children.2) row1567_eq_data branches
set_option maxRecDepth 4096 in
theorem row1579_eq_data : row1579 = Data.row1579 := rfl
def node1579 : Graph := .node 1579 row1579 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1579_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1579 :=
  Budget.node_covers _ _ 1579 row1579 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1579_covers
theorem node1579_eq_data : node1579 = Data.node1579 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1579 row children.1 children.2) row1579_eq_data branches
set_option maxRecDepth 4096 in
theorem row1583_eq_data : row1583 = Data.row1583 := rfl
def node1583 : Graph := .node 1583 row1583 node1579 Graph.empty
set_option maxRecDepth 4096 in
theorem node1583_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1583 :=
  Budget.node_covers _ _ 1583 row1583 node1579 Graph.empty
    node1579_covers empty_covers (by decide) (by decide) row1583_covers
theorem node1583_eq_data : node1583 = Data.node1583 := by
  have branches : (node1579, Graph.empty) = (Data.node1579, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1579_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1583 row children.1 children.2) row1583_eq_data branches
set_option maxRecDepth 4096 in
theorem row1571_eq_data : row1571 = Data.row1571 := rfl
def node1571 : Graph := .node 1571 row1571 node1567 node1583
set_option maxRecDepth 4096 in
theorem node1571_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1571 :=
  Budget.node_covers _ _ 1571 row1571 node1567 node1583
    node1567_covers node1583_covers (by decide) (by decide) row1571_covers
theorem node1571_eq_data : node1571 = Data.node1571 := by
  have branches : (node1567, node1583) = (Data.node1567, Data.node1583) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1567_eq_data node1583_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1571 row children.1 children.2) row1571_eq_data branches
set_option maxRecDepth 4096 in
theorem row1601_eq_data : row1601 = Data.row1601 := rfl
def node1601 : Graph := .node 1601 row1601 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1601_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1601 :=
  Budget.node_covers _ _ 1601 row1601 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1601_covers
theorem node1601_eq_data : node1601 = Data.node1601 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1601 row children.1 children.2) row1601_eq_data branches
set_option maxRecDepth 4096 in
theorem row1607_eq_data : row1607 = Data.row1607 := rfl
def node1607 : Graph := .node 1607 row1607 node1601 Graph.empty
set_option maxRecDepth 4096 in
theorem node1607_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1607 :=
  Budget.node_covers _ _ 1607 row1607 node1601 Graph.empty
    node1601_covers empty_covers (by decide) (by decide) row1607_covers
theorem node1607_eq_data : node1607 = Data.node1607 := by
  have branches : (node1601, Graph.empty) = (Data.node1601, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1601_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1607 row children.1 children.2) row1607_eq_data branches
set_option maxRecDepth 4096 in
theorem row1613_eq_data : row1613 = Data.row1613 := rfl
def node1613 : Graph := .node 1613 row1613 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1613_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1613 :=
  Budget.node_covers _ _ 1613 row1613 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1613_covers
theorem node1613_eq_data : node1613 = Data.node1613 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1613 row children.1 children.2) row1613_eq_data branches
set_option maxRecDepth 4096 in
theorem row1619_eq_data : row1619 = Data.row1619 := rfl
def node1619 : Graph := .node 1619 row1619 node1613 Graph.empty
set_option maxRecDepth 4096 in
theorem node1619_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1619 :=
  Budget.node_covers _ _ 1619 row1619 node1613 Graph.empty
    node1613_covers empty_covers (by decide) (by decide) row1619_covers
theorem node1619_eq_data : node1619 = Data.node1619 := by
  have branches : (node1613, Graph.empty) = (Data.node1613, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1613_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1619 row children.1 children.2) row1619_eq_data branches
set_option maxRecDepth 4096 in
theorem row1609_eq_data : row1609 = Data.row1609 := rfl
def node1609 : Graph := .node 1609 row1609 node1607 node1619
set_option maxRecDepth 4096 in
theorem node1609_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1609 :=
  Budget.node_covers _ _ 1609 row1609 node1607 node1619
    node1607_covers node1619_covers (by decide) (by decide) row1609_covers
theorem node1609_eq_data : node1609 = Data.node1609 := by
  have branches : (node1607, node1619) = (Data.node1607, Data.node1619) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1607_eq_data node1619_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1609 row children.1 children.2) row1609_eq_data branches
set_option maxRecDepth 4096 in
theorem row1597_eq_data : row1597 = Data.row1597 := rfl
def node1597 : Graph := .node 1597 row1597 node1571 node1609
set_option maxRecDepth 4096 in
theorem node1597_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1597 :=
  Budget.node_covers _ _ 1597 row1597 node1571 node1609
    node1571_covers node1609_covers (by decide) (by decide) row1597_covers
theorem node1597_eq_data : node1597 = Data.node1597 := by
  have branches : (node1571, node1609) = (Data.node1571, Data.node1609) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1571_eq_data node1609_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1597 row children.1 children.2) row1597_eq_data branches
set_option maxRecDepth 4096 in
theorem row1627_eq_data : row1627 = Data.row1627 := rfl
def node1627 : Graph := .node 1627 row1627 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1627_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1627 :=
  Budget.node_covers _ _ 1627 row1627 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1627_covers
theorem node1627_eq_data : node1627 = Data.node1627 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1627 row children.1 children.2) row1627_eq_data branches
set_option maxRecDepth 4096 in
theorem row1637_eq_data : row1637 = Data.row1637 := rfl
def node1637 : Graph := .node 1637 row1637 node1627 Graph.empty
set_option maxRecDepth 4096 in
theorem node1637_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1637 :=
  Budget.node_covers _ _ 1637 row1637 node1627 Graph.empty
    node1627_covers empty_covers (by decide) (by decide) row1637_covers
theorem node1637_eq_data : node1637 = Data.node1637 := by
  have branches : (node1627, Graph.empty) = (Data.node1627, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1627_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1637 row children.1 children.2) row1637_eq_data branches
set_option maxRecDepth 4096 in
theorem row1663_eq_data : row1663 = Data.row1663 := rfl
def node1663 : Graph := .node 1663 row1663 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1663_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1663 :=
  Budget.node_covers _ _ 1663 row1663 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1663_covers
theorem node1663_eq_data : node1663 = Data.node1663 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1663 row children.1 children.2) row1663_eq_data branches
set_option maxRecDepth 4096 in
theorem row1667_eq_data : row1667 = Data.row1667 := rfl
def node1667 : Graph := .node 1667 row1667 node1663 Graph.empty
set_option maxRecDepth 4096 in
theorem node1667_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1667 :=
  Budget.node_covers _ _ 1667 row1667 node1663 Graph.empty
    node1663_covers empty_covers (by decide) (by decide) row1667_covers
theorem node1667_eq_data : node1667 = Data.node1667 := by
  have branches : (node1663, Graph.empty) = (Data.node1663, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1663_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1667 row children.1 children.2) row1667_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
