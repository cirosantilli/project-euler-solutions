import ProjectEulerSolutions.Lib.PrimePairLarge.Graph008
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row4157_eq_data : row4157 = Data.row4157 := rfl
def node4157 : Graph := .node 4157 row4157 node4111 node4219
set_option maxRecDepth 4096 in
theorem node4157_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4157 :=
  Budget.node_covers _ _ 4157 row4157 node4111 node4219
    node4111_covers node4219_covers (by decide) (by decide) row4157_covers
theorem node4157_eq_data : node4157 = Data.node4157 := by
  have branches : (node4111, node4219) = (Data.node4111, Data.node4219) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4111_eq_data node4219_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4157 row children.1 children.2) row4157_eq_data branches
set_option maxRecDepth 4096 in
theorem row4057_eq_data : row4057 = Data.row4057 := rfl
def node4057 : Graph := .node 4057 row4057 node3947 node4157
set_option maxRecDepth 4096 in
theorem node4057_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4057 :=
  Budget.node_covers _ _ 4057 row4057 node3947 node4157
    node3947_covers node4157_covers (by decide) (by decide) row4057_covers
theorem node4057_eq_data : node4057 = Data.node4057 := by
  have branches : (node3947, node4157) = (Data.node3947, Data.node4157) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3947_eq_data node4157_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4057 row children.1 children.2) row4057_eq_data branches
set_option maxRecDepth 4096 in
theorem row3863_eq_data : row3863 = Data.row3863 := rfl
def node3863 : Graph := .node 3863 row3863 node3671 node4057
set_option maxRecDepth 4096 in
theorem node3863_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3863 :=
  Budget.node_covers _ _ 3863 row3863 node3671 node4057
    node3671_covers node4057_covers (by decide) (by decide) row3863_covers
theorem node3863_eq_data : node3863 = Data.node3863 := by
  have branches : (node3671, node4057) = (Data.node3671, Data.node4057) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3671_eq_data node4057_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3863 row children.1 children.2) row3863_eq_data branches
set_option maxRecDepth 4096 in
theorem row3469_eq_data : row3469 = Data.row3469 := rfl
def node3469 : Graph := .node 3469 row3469 node3067 node3863
set_option maxRecDepth 4096 in
theorem node3469_covers : Budget.Covers PrimePairVertices.largeTree 26034 node3469 :=
  Budget.node_covers _ _ 3469 row3469 node3067 node3863
    node3067_covers node3863_covers (by decide) (by decide) row3469_covers
theorem node3469_eq_data : node3469 = Data.node3469 := by
  have branches : (node3067, node3863) = (Data.node3067, Data.node3863) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node3067_eq_data node3863_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 3469 row children.1 children.2) row3469_eq_data branches
set_option maxRecDepth 4096 in
theorem row4261_eq_data : row4261 = Data.row4261 := rfl
def node4261 : Graph := .node 4261 row4261 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4261_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4261 :=
  Budget.node_covers _ _ 4261 row4261 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4261_covers
theorem node4261_eq_data : node4261 = Data.node4261 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4261 row children.1 children.2) row4261_eq_data branches
set_option maxRecDepth 4096 in
theorem row4273_eq_data : row4273 = Data.row4273 := rfl
def node4273 : Graph := .node 4273 row4273 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4273_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4273 :=
  Budget.node_covers _ _ 4273 row4273 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4273_covers
theorem node4273_eq_data : node4273 = Data.node4273 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4273 row children.1 children.2) row4273_eq_data branches
set_option maxRecDepth 4096 in
theorem row4271_eq_data : row4271 = Data.row4271 := rfl
def node4271 : Graph := .node 4271 row4271 node4261 node4273
set_option maxRecDepth 4096 in
theorem node4271_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4271 :=
  Budget.node_covers _ _ 4271 row4271 node4261 node4273
    node4261_covers node4273_covers (by decide) (by decide) row4271_covers
theorem node4271_eq_data : node4271 = Data.node4271 := by
  have branches : (node4261, node4273) = (Data.node4261, Data.node4273) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4261_eq_data node4273_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4271 row children.1 children.2) row4271_eq_data branches
set_option maxRecDepth 4096 in
theorem row4289_eq_data : row4289 = Data.row4289 := rfl
def node4289 : Graph := .node 4289 row4289 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4289_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4289 :=
  Budget.node_covers _ _ 4289 row4289 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4289_covers
theorem node4289_eq_data : node4289 = Data.node4289 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4289 row children.1 children.2) row4289_eq_data branches
set_option maxRecDepth 4096 in
theorem row4297_eq_data : row4297 = Data.row4297 := rfl
def node4297 : Graph := .node 4297 row4297 node4289 Graph.empty
set_option maxRecDepth 4096 in
theorem node4297_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4297 :=
  Budget.node_covers _ _ 4297 row4297 node4289 Graph.empty
    node4289_covers empty_covers (by decide) (by decide) row4297_covers
theorem node4297_eq_data : node4297 = Data.node4297 := by
  have branches : (node4289, Graph.empty) = (Data.node4289, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4289_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4297 row children.1 children.2) row4297_eq_data branches
set_option maxRecDepth 4096 in
theorem row4283_eq_data : row4283 = Data.row4283 := rfl
def node4283 : Graph := .node 4283 row4283 node4271 node4297
set_option maxRecDepth 4096 in
theorem node4283_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4283 :=
  Budget.node_covers _ _ 4283 row4283 node4271 node4297
    node4271_covers node4297_covers (by decide) (by decide) row4283_covers
theorem node4283_eq_data : node4283 = Data.node4283 := by
  have branches : (node4271, node4297) = (Data.node4271, Data.node4297) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4271_eq_data node4297_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4283 row children.1 children.2) row4283_eq_data branches
set_option maxRecDepth 4096 in
theorem row4337_eq_data : row4337 = Data.row4337 := rfl
def node4337 : Graph := .node 4337 row4337 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4337_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4337 :=
  Budget.node_covers _ _ 4337 row4337 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4337_covers
theorem node4337_eq_data : node4337 = Data.node4337 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4337 row children.1 children.2) row4337_eq_data branches
set_option maxRecDepth 4096 in
theorem row4339_eq_data : row4339 = Data.row4339 := rfl
def node4339 : Graph := .node 4339 row4339 node4337 Graph.empty
set_option maxRecDepth 4096 in
theorem node4339_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4339 :=
  Budget.node_covers _ _ 4339 row4339 node4337 Graph.empty
    node4337_covers empty_covers (by decide) (by decide) row4339_covers
theorem node4339_eq_data : node4339 = Data.node4339 := by
  have branches : (node4337, Graph.empty) = (Data.node4337, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4337_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4339 row children.1 children.2) row4339_eq_data branches
set_option maxRecDepth 4096 in
theorem row4357_eq_data : row4357 = Data.row4357 := rfl
def node4357 : Graph := .node 4357 row4357 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4357_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4357 :=
  Budget.node_covers _ _ 4357 row4357 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4357_covers
theorem node4357_eq_data : node4357 = Data.node4357 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4357 row children.1 children.2) row4357_eq_data branches
set_option maxRecDepth 4096 in
theorem row4363_eq_data : row4363 = Data.row4363 := rfl
def node4363 : Graph := .node 4363 row4363 node4357 Graph.empty
set_option maxRecDepth 4096 in
theorem node4363_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4363 :=
  Budget.node_covers _ _ 4363 row4363 node4357 Graph.empty
    node4357_covers empty_covers (by decide) (by decide) row4363_covers
theorem node4363_eq_data : node4363 = Data.node4363 := by
  have branches : (node4357, Graph.empty) = (Data.node4357, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4357_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4363 row children.1 children.2) row4363_eq_data branches
set_option maxRecDepth 4096 in
theorem row4349_eq_data : row4349 = Data.row4349 := rfl
def node4349 : Graph := .node 4349 row4349 node4339 node4363
set_option maxRecDepth 4096 in
theorem node4349_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4349 :=
  Budget.node_covers _ _ 4349 row4349 node4339 node4363
    node4339_covers node4363_covers (by decide) (by decide) row4349_covers
theorem node4349_eq_data : node4349 = Data.node4349 := by
  have branches : (node4339, node4363) = (Data.node4339, Data.node4363) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4339_eq_data node4363_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4349 row children.1 children.2) row4349_eq_data branches
set_option maxRecDepth 4096 in
theorem row4327_eq_data : row4327 = Data.row4327 := rfl
def node4327 : Graph := .node 4327 row4327 node4283 node4349
set_option maxRecDepth 4096 in
theorem node4327_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4327 :=
  Budget.node_covers _ _ 4327 row4327 node4283 node4349
    node4283_covers node4349_covers (by decide) (by decide) row4327_covers
theorem node4327_eq_data : node4327 = Data.node4327 := by
  have branches : (node4283, node4349) = (Data.node4283, Data.node4349) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4283_eq_data node4349_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4327 row children.1 children.2) row4327_eq_data branches
set_option maxRecDepth 4096 in
theorem row4391_eq_data : row4391 = Data.row4391 := rfl
def node4391 : Graph := .node 4391 row4391 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4391_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4391 :=
  Budget.node_covers _ _ 4391 row4391 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4391_covers
theorem node4391_eq_data : node4391 = Data.node4391 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4391 row children.1 children.2) row4391_eq_data branches
set_option maxRecDepth 4096 in
theorem row4397_eq_data : row4397 = Data.row4397 := rfl
def node4397 : Graph := .node 4397 row4397 node4391 Graph.empty
set_option maxRecDepth 4096 in
theorem node4397_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4397 :=
  Budget.node_covers _ _ 4397 row4397 node4391 Graph.empty
    node4391_covers empty_covers (by decide) (by decide) row4397_covers
theorem node4397_eq_data : node4397 = Data.node4397 := by
  have branches : (node4391, Graph.empty) = (Data.node4391, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4391_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4397 row children.1 children.2) row4397_eq_data branches
set_option maxRecDepth 4096 in
theorem row4421_eq_data : row4421 = Data.row4421 := rfl
def node4421 : Graph := .node 4421 row4421 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4421_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4421 :=
  Budget.node_covers _ _ 4421 row4421 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4421_covers
theorem node4421_eq_data : node4421 = Data.node4421 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4421 row children.1 children.2) row4421_eq_data branches
set_option maxRecDepth 4096 in
theorem row4423_eq_data : row4423 = Data.row4423 := rfl
def node4423 : Graph := .node 4423 row4423 node4421 Graph.empty
set_option maxRecDepth 4096 in
theorem node4423_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4423 :=
  Budget.node_covers _ _ 4423 row4423 node4421 Graph.empty
    node4421_covers empty_covers (by decide) (by decide) row4423_covers
theorem node4423_eq_data : node4423 = Data.node4423 := by
  have branches : (node4421, Graph.empty) = (Data.node4421, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4421_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4423 row children.1 children.2) row4423_eq_data branches
set_option maxRecDepth 4096 in
theorem row4409_eq_data : row4409 = Data.row4409 := rfl
def node4409 : Graph := .node 4409 row4409 node4397 node4423
set_option maxRecDepth 4096 in
theorem node4409_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4409 :=
  Budget.node_covers _ _ 4409 row4409 node4397 node4423
    node4397_covers node4423_covers (by decide) (by decide) row4409_covers
theorem node4409_eq_data : node4409 = Data.node4409 := by
  have branches : (node4397, node4423) = (Data.node4397, Data.node4423) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4397_eq_data node4423_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4409 row children.1 children.2) row4409_eq_data branches
set_option maxRecDepth 4096 in
theorem row4447_eq_data : row4447 = Data.row4447 := rfl
def node4447 : Graph := .node 4447 row4447 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4447_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4447 :=
  Budget.node_covers _ _ 4447 row4447 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4447_covers
theorem node4447_eq_data : node4447 = Data.node4447 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4447 row children.1 children.2) row4447_eq_data branches
set_option maxRecDepth 4096 in
theorem row4451_eq_data : row4451 = Data.row4451 := rfl
def node4451 : Graph := .node 4451 row4451 node4447 Graph.empty
set_option maxRecDepth 4096 in
theorem node4451_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4451 :=
  Budget.node_covers _ _ 4451 row4451 node4447 Graph.empty
    node4447_covers empty_covers (by decide) (by decide) row4451_covers
theorem node4451_eq_data : node4451 = Data.node4451 := by
  have branches : (node4447, Graph.empty) = (Data.node4447, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4447_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4451 row children.1 children.2) row4451_eq_data branches
set_option maxRecDepth 4096 in
theorem row4463_eq_data : row4463 = Data.row4463 := rfl
def node4463 : Graph := .node 4463 row4463 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4463_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4463 :=
  Budget.node_covers _ _ 4463 row4463 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4463_covers
theorem node4463_eq_data : node4463 = Data.node4463 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4463 row children.1 children.2) row4463_eq_data branches
set_option maxRecDepth 4096 in
theorem row4481_eq_data : row4481 = Data.row4481 := rfl
def node4481 : Graph := .node 4481 row4481 node4463 Graph.empty
set_option maxRecDepth 4096 in
theorem node4481_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4481 :=
  Budget.node_covers _ _ 4481 row4481 node4463 Graph.empty
    node4463_covers empty_covers (by decide) (by decide) row4481_covers
theorem node4481_eq_data : node4481 = Data.node4481 := by
  have branches : (node4463, Graph.empty) = (Data.node4463, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4463_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4481 row children.1 children.2) row4481_eq_data branches
set_option maxRecDepth 4096 in
theorem row4457_eq_data : row4457 = Data.row4457 := rfl
def node4457 : Graph := .node 4457 row4457 node4451 node4481
set_option maxRecDepth 4096 in
theorem node4457_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4457 :=
  Budget.node_covers _ _ 4457 row4457 node4451 node4481
    node4451_covers node4481_covers (by decide) (by decide) row4457_covers
theorem node4457_eq_data : node4457 = Data.node4457 := by
  have branches : (node4451, node4481) = (Data.node4451, Data.node4481) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4451_eq_data node4481_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4457 row children.1 children.2) row4457_eq_data branches
set_option maxRecDepth 4096 in
theorem row4441_eq_data : row4441 = Data.row4441 := rfl
def node4441 : Graph := .node 4441 row4441 node4409 node4457
set_option maxRecDepth 4096 in
theorem node4441_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4441 :=
  Budget.node_covers _ _ 4441 row4441 node4409 node4457
    node4409_covers node4457_covers (by decide) (by decide) row4441_covers
theorem node4441_eq_data : node4441 = Data.node4441 := by
  have branches : (node4409, node4457) = (Data.node4409, Data.node4457) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4409_eq_data node4457_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4441 row children.1 children.2) row4441_eq_data branches
set_option maxRecDepth 4096 in
theorem row4373_eq_data : row4373 = Data.row4373 := rfl
def node4373 : Graph := .node 4373 row4373 node4327 node4441
set_option maxRecDepth 4096 in
theorem node4373_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4373 :=
  Budget.node_covers _ _ 4373 row4373 node4327 node4441
    node4327_covers node4441_covers (by decide) (by decide) row4373_covers
theorem node4373_eq_data : node4373 = Data.node4373 := by
  have branches : (node4327, node4441) = (Data.node4327, Data.node4441) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4327_eq_data node4441_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4373 row children.1 children.2) row4373_eq_data branches
set_option maxRecDepth 4096 in
theorem row4493_eq_data : row4493 = Data.row4493 := rfl
def node4493 : Graph := .node 4493 row4493 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4493_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4493 :=
  Budget.node_covers _ _ 4493 row4493 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4493_covers
theorem node4493_eq_data : node4493 = Data.node4493 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4493 row children.1 children.2) row4493_eq_data branches
set_option maxRecDepth 4096 in
theorem row4507_eq_data : row4507 = Data.row4507 := rfl
def node4507 : Graph := .node 4507 row4507 node4493 Graph.empty
set_option maxRecDepth 4096 in
theorem node4507_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4507 :=
  Budget.node_covers _ _ 4507 row4507 node4493 Graph.empty
    node4493_covers empty_covers (by decide) (by decide) row4507_covers
theorem node4507_eq_data : node4507 = Data.node4507 := by
  have branches : (node4493, Graph.empty) = (Data.node4493, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4493_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4507 row children.1 children.2) row4507_eq_data branches
set_option maxRecDepth 4096 in
theorem row4517_eq_data : row4517 = Data.row4517 := rfl
def node4517 : Graph := .node 4517 row4517 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4517_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4517 :=
  Budget.node_covers _ _ 4517 row4517 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4517_covers
theorem node4517_eq_data : node4517 = Data.node4517 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4517 row children.1 children.2) row4517_eq_data branches
set_option maxRecDepth 4096 in
theorem row4519_eq_data : row4519 = Data.row4519 := rfl
def node4519 : Graph := .node 4519 row4519 node4517 Graph.empty
set_option maxRecDepth 4096 in
theorem node4519_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4519 :=
  Budget.node_covers _ _ 4519 row4519 node4517 Graph.empty
    node4517_covers empty_covers (by decide) (by decide) row4519_covers
theorem node4519_eq_data : node4519 = Data.node4519 := by
  have branches : (node4517, Graph.empty) = (Data.node4517, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4517_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4519 row children.1 children.2) row4519_eq_data branches
set_option maxRecDepth 4096 in
theorem row4513_eq_data : row4513 = Data.row4513 := rfl
def node4513 : Graph := .node 4513 row4513 node4507 node4519
set_option maxRecDepth 4096 in
theorem node4513_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4513 :=
  Budget.node_covers _ _ 4513 row4513 node4507 node4519
    node4507_covers node4519_covers (by decide) (by decide) row4513_covers
theorem node4513_eq_data : node4513 = Data.node4513 := by
  have branches : (node4507, node4519) = (Data.node4507, Data.node4519) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4507_eq_data node4519_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4513 row children.1 children.2) row4513_eq_data branches
set_option maxRecDepth 4096 in
theorem row4547_eq_data : row4547 = Data.row4547 := rfl
def node4547 : Graph := .node 4547 row4547 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4547_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4547 :=
  Budget.node_covers _ _ 4547 row4547 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4547_covers
theorem node4547_eq_data : node4547 = Data.node4547 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4547 row children.1 children.2) row4547_eq_data branches
set_option maxRecDepth 4096 in
theorem row4549_eq_data : row4549 = Data.row4549 := rfl
def node4549 : Graph := .node 4549 row4549 node4547 Graph.empty
set_option maxRecDepth 4096 in
theorem node4549_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4549 :=
  Budget.node_covers _ _ 4549 row4549 node4547 Graph.empty
    node4547_covers empty_covers (by decide) (by decide) row4549_covers
theorem node4549_eq_data : node4549 = Data.node4549 := by
  have branches : (node4547, Graph.empty) = (Data.node4547, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4547_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4549 row children.1 children.2) row4549_eq_data branches
set_option maxRecDepth 4096 in
theorem row4567_eq_data : row4567 = Data.row4567 := rfl
def node4567 : Graph := .node 4567 row4567 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4567_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4567 :=
  Budget.node_covers _ _ 4567 row4567 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4567_covers
theorem node4567_eq_data : node4567 = Data.node4567 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4567 row children.1 children.2) row4567_eq_data branches
set_option maxRecDepth 4096 in
theorem row4583_eq_data : row4583 = Data.row4583 := rfl
def node4583 : Graph := .node 4583 row4583 node4567 Graph.empty
set_option maxRecDepth 4096 in
theorem node4583_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4583 :=
  Budget.node_covers _ _ 4583 row4583 node4567 Graph.empty
    node4567_covers empty_covers (by decide) (by decide) row4583_covers
theorem node4583_eq_data : node4583 = Data.node4583 := by
  have branches : (node4567, Graph.empty) = (Data.node4567, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4567_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4583 row children.1 children.2) row4583_eq_data branches
set_option maxRecDepth 4096 in
theorem row4561_eq_data : row4561 = Data.row4561 := rfl
def node4561 : Graph := .node 4561 row4561 node4549 node4583
set_option maxRecDepth 4096 in
theorem node4561_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4561 :=
  Budget.node_covers _ _ 4561 row4561 node4549 node4583
    node4549_covers node4583_covers (by decide) (by decide) row4561_covers
theorem node4561_eq_data : node4561 = Data.node4561 := by
  have branches : (node4549, node4583) = (Data.node4549, Data.node4583) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4549_eq_data node4583_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4561 row children.1 children.2) row4561_eq_data branches
set_option maxRecDepth 4096 in
theorem row4523_eq_data : row4523 = Data.row4523 := rfl
def node4523 : Graph := .node 4523 row4523 node4513 node4561
set_option maxRecDepth 4096 in
theorem node4523_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4523 :=
  Budget.node_covers _ _ 4523 row4523 node4513 node4561
    node4513_covers node4561_covers (by decide) (by decide) row4523_covers
theorem node4523_eq_data : node4523 = Data.node4523 := by
  have branches : (node4513, node4561) = (Data.node4513, Data.node4561) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4513_eq_data node4561_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4523 row children.1 children.2) row4523_eq_data branches
set_option maxRecDepth 4096 in
theorem row4597_eq_data : row4597 = Data.row4597 := rfl
def node4597 : Graph := .node 4597 row4597 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4597_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4597 :=
  Budget.node_covers _ _ 4597 row4597 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4597_covers
theorem node4597_eq_data : node4597 = Data.node4597 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4597 row children.1 children.2) row4597_eq_data branches
set_option maxRecDepth 4096 in
theorem row4603_eq_data : row4603 = Data.row4603 := rfl
def node4603 : Graph := .node 4603 row4603 node4597 Graph.empty
set_option maxRecDepth 4096 in
theorem node4603_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4603 :=
  Budget.node_covers _ _ 4603 row4603 node4597 Graph.empty
    node4597_covers empty_covers (by decide) (by decide) row4603_covers
theorem node4603_eq_data : node4603 = Data.node4603 := by
  have branches : (node4597, Graph.empty) = (Data.node4597, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4597_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4603 row children.1 children.2) row4603_eq_data branches
set_option maxRecDepth 4096 in
theorem row4637_eq_data : row4637 = Data.row4637 := rfl
def node4637 : Graph := .node 4637 row4637 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4637_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4637 :=
  Budget.node_covers _ _ 4637 row4637 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4637_covers
theorem node4637_eq_data : node4637 = Data.node4637 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4637 row children.1 children.2) row4637_eq_data branches
set_option maxRecDepth 4096 in
theorem row4639_eq_data : row4639 = Data.row4639 := rfl
def node4639 : Graph := .node 4639 row4639 node4637 Graph.empty
set_option maxRecDepth 4096 in
theorem node4639_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4639 :=
  Budget.node_covers _ _ 4639 row4639 node4637 Graph.empty
    node4637_covers empty_covers (by decide) (by decide) row4639_covers
theorem node4639_eq_data : node4639 = Data.node4639 := by
  have branches : (node4637, Graph.empty) = (Data.node4637, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4637_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4639 row children.1 children.2) row4639_eq_data branches
set_option maxRecDepth 4096 in
theorem row4621_eq_data : row4621 = Data.row4621 := rfl
def node4621 : Graph := .node 4621 row4621 node4603 node4639
set_option maxRecDepth 4096 in
theorem node4621_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4621 :=
  Budget.node_covers _ _ 4621 row4621 node4603 node4639
    node4603_covers node4639_covers (by decide) (by decide) row4621_covers
theorem node4621_eq_data : node4621 = Data.node4621 := by
  have branches : (node4603, node4639) = (Data.node4603, Data.node4639) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4603_eq_data node4639_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4621 row children.1 children.2) row4621_eq_data branches
set_option maxRecDepth 4096 in
theorem row4649_eq_data : row4649 = Data.row4649 := rfl
def node4649 : Graph := .node 4649 row4649 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4649_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4649 :=
  Budget.node_covers _ _ 4649 row4649 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4649_covers
theorem node4649_eq_data : node4649 = Data.node4649 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4649 row children.1 children.2) row4649_eq_data branches
set_option maxRecDepth 4096 in
theorem row4651_eq_data : row4651 = Data.row4651 := rfl
def node4651 : Graph := .node 4651 row4651 node4649 Graph.empty
set_option maxRecDepth 4096 in
theorem node4651_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4651 :=
  Budget.node_covers _ _ 4651 row4651 node4649 Graph.empty
    node4649_covers empty_covers (by decide) (by decide) row4651_covers
theorem node4651_eq_data : node4651 = Data.node4651 := by
  have branches : (node4649, Graph.empty) = (Data.node4649, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4649_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4651 row children.1 children.2) row4651_eq_data branches
set_option maxRecDepth 4096 in
theorem row4663_eq_data : row4663 = Data.row4663 := rfl
def node4663 : Graph := .node 4663 row4663 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4663_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4663 :=
  Budget.node_covers _ _ 4663 row4663 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4663_covers
theorem node4663_eq_data : node4663 = Data.node4663 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4663 row children.1 children.2) row4663_eq_data branches
set_option maxRecDepth 4096 in
theorem row4673_eq_data : row4673 = Data.row4673 := rfl
def node4673 : Graph := .node 4673 row4673 node4663 Graph.empty
set_option maxRecDepth 4096 in
theorem node4673_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4673 :=
  Budget.node_covers _ _ 4673 row4673 node4663 Graph.empty
    node4663_covers empty_covers (by decide) (by decide) row4673_covers
theorem node4673_eq_data : node4673 = Data.node4673 := by
  have branches : (node4663, Graph.empty) = (Data.node4663, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4663_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4673 row children.1 children.2) row4673_eq_data branches
set_option maxRecDepth 4096 in
theorem row4657_eq_data : row4657 = Data.row4657 := rfl
def node4657 : Graph := .node 4657 row4657 node4651 node4673
set_option maxRecDepth 4096 in
theorem node4657_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4657 :=
  Budget.node_covers _ _ 4657 row4657 node4651 node4673
    node4651_covers node4673_covers (by decide) (by decide) row4657_covers
theorem node4657_eq_data : node4657 = Data.node4657 := by
  have branches : (node4651, node4673) = (Data.node4651, Data.node4673) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4651_eq_data node4673_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4657 row children.1 children.2) row4657_eq_data branches
set_option maxRecDepth 4096 in
theorem row4643_eq_data : row4643 = Data.row4643 := rfl
def node4643 : Graph := .node 4643 row4643 node4621 node4657
set_option maxRecDepth 4096 in
theorem node4643_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4643 :=
  Budget.node_covers _ _ 4643 row4643 node4621 node4657
    node4621_covers node4657_covers (by decide) (by decide) row4643_covers
theorem node4643_eq_data : node4643 = Data.node4643 := by
  have branches : (node4621, node4657) = (Data.node4621, Data.node4657) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4621_eq_data node4657_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4643 row children.1 children.2) row4643_eq_data branches
set_option maxRecDepth 4096 in
theorem row4591_eq_data : row4591 = Data.row4591 := rfl
def node4591 : Graph := .node 4591 row4591 node4523 node4643
set_option maxRecDepth 4096 in
theorem node4591_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4591 :=
  Budget.node_covers _ _ 4591 row4591 node4523 node4643
    node4523_covers node4643_covers (by decide) (by decide) row4591_covers
theorem node4591_eq_data : node4591 = Data.node4591 := by
  have branches : (node4523, node4643) = (Data.node4523, Data.node4643) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4523_eq_data node4643_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4591 row children.1 children.2) row4591_eq_data branches
set_option maxRecDepth 4096 in
theorem row4483_eq_data : row4483 = Data.row4483 := rfl
def node4483 : Graph := .node 4483 row4483 node4373 node4591
set_option maxRecDepth 4096 in
theorem node4483_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4483 :=
  Budget.node_covers _ _ 4483 row4483 node4373 node4591
    node4373_covers node4591_covers (by decide) (by decide) row4483_covers
theorem node4483_eq_data : node4483 = Data.node4483 := by
  have branches : (node4373, node4591) = (Data.node4373, Data.node4591) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4373_eq_data node4591_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4483 row children.1 children.2) row4483_eq_data branches
set_option maxRecDepth 4096 in
theorem row4691_eq_data : row4691 = Data.row4691 := rfl
def node4691 : Graph := .node 4691 row4691 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4691_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4691 :=
  Budget.node_covers _ _ 4691 row4691 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4691_covers
theorem node4691_eq_data : node4691 = Data.node4691 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4691 row children.1 children.2) row4691_eq_data branches
set_option maxRecDepth 4096 in
theorem row4703_eq_data : row4703 = Data.row4703 := rfl
def node4703 : Graph := .node 4703 row4703 node4691 Graph.empty
set_option maxRecDepth 4096 in
theorem node4703_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4703 :=
  Budget.node_covers _ _ 4703 row4703 node4691 Graph.empty
    node4691_covers empty_covers (by decide) (by decide) row4703_covers
theorem node4703_eq_data : node4703 = Data.node4703 := by
  have branches : (node4691, Graph.empty) = (Data.node4691, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4691_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4703 row children.1 children.2) row4703_eq_data branches
set_option maxRecDepth 4096 in
theorem row4723_eq_data : row4723 = Data.row4723 := rfl
def node4723 : Graph := .node 4723 row4723 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4723_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4723 :=
  Budget.node_covers _ _ 4723 row4723 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4723_covers
theorem node4723_eq_data : node4723 = Data.node4723 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4723 row children.1 children.2) row4723_eq_data branches
set_option maxRecDepth 4096 in
theorem row4729_eq_data : row4729 = Data.row4729 := rfl
def node4729 : Graph := .node 4729 row4729 node4723 Graph.empty
set_option maxRecDepth 4096 in
theorem node4729_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4729 :=
  Budget.node_covers _ _ 4729 row4729 node4723 Graph.empty
    node4723_covers empty_covers (by decide) (by decide) row4729_covers
theorem node4729_eq_data : node4729 = Data.node4729 := by
  have branches : (node4723, Graph.empty) = (Data.node4723, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4723_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4729 row children.1 children.2) row4729_eq_data branches
set_option maxRecDepth 4096 in
theorem row4721_eq_data : row4721 = Data.row4721 := rfl
def node4721 : Graph := .node 4721 row4721 node4703 node4729
set_option maxRecDepth 4096 in
theorem node4721_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4721 :=
  Budget.node_covers _ _ 4721 row4721 node4703 node4729
    node4703_covers node4729_covers (by decide) (by decide) row4721_covers
theorem node4721_eq_data : node4721 = Data.node4721 := by
  have branches : (node4703, node4729) = (Data.node4703, Data.node4729) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4703_eq_data node4729_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4721 row children.1 children.2) row4721_eq_data branches
set_option maxRecDepth 4096 in
theorem row4751_eq_data : row4751 = Data.row4751 := rfl
def node4751 : Graph := .node 4751 row4751 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4751_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4751 :=
  Budget.node_covers _ _ 4751 row4751 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4751_covers
theorem node4751_eq_data : node4751 = Data.node4751 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4751 row children.1 children.2) row4751_eq_data branches
set_option maxRecDepth 4096 in
theorem row4759_eq_data : row4759 = Data.row4759 := rfl
def node4759 : Graph := .node 4759 row4759 node4751 Graph.empty
set_option maxRecDepth 4096 in
theorem node4759_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4759 :=
  Budget.node_covers _ _ 4759 row4759 node4751 Graph.empty
    node4751_covers empty_covers (by decide) (by decide) row4759_covers
theorem node4759_eq_data : node4759 = Data.node4759 := by
  have branches : (node4751, Graph.empty) = (Data.node4751, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4751_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4759 row children.1 children.2) row4759_eq_data branches
set_option maxRecDepth 4096 in
theorem row4787_eq_data : row4787 = Data.row4787 := rfl
def node4787 : Graph := .node 4787 row4787 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4787_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4787 :=
  Budget.node_covers _ _ 4787 row4787 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4787_covers
theorem node4787_eq_data : node4787 = Data.node4787 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4787 row children.1 children.2) row4787_eq_data branches
set_option maxRecDepth 4096 in
theorem row4789_eq_data : row4789 = Data.row4789 := rfl
def node4789 : Graph := .node 4789 row4789 node4787 Graph.empty
set_option maxRecDepth 4096 in
theorem node4789_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4789 :=
  Budget.node_covers _ _ 4789 row4789 node4787 Graph.empty
    node4787_covers empty_covers (by decide) (by decide) row4789_covers
theorem node4789_eq_data : node4789 = Data.node4789 := by
  have branches : (node4787, Graph.empty) = (Data.node4787, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4787_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4789 row children.1 children.2) row4789_eq_data branches
set_option maxRecDepth 4096 in
theorem row4783_eq_data : row4783 = Data.row4783 := rfl
def node4783 : Graph := .node 4783 row4783 node4759 node4789
set_option maxRecDepth 4096 in
theorem node4783_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4783 :=
  Budget.node_covers _ _ 4783 row4783 node4759 node4789
    node4759_covers node4789_covers (by decide) (by decide) row4783_covers
theorem node4783_eq_data : node4783 = Data.node4783 := by
  have branches : (node4759, node4789) = (Data.node4759, Data.node4789) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4759_eq_data node4789_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4783 row children.1 children.2) row4783_eq_data branches
set_option maxRecDepth 4096 in
theorem row4733_eq_data : row4733 = Data.row4733 := rfl
def node4733 : Graph := .node 4733 row4733 node4721 node4783
set_option maxRecDepth 4096 in
theorem node4733_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4733 :=
  Budget.node_covers _ _ 4733 row4733 node4721 node4783
    node4721_covers node4783_covers (by decide) (by decide) row4733_covers
theorem node4733_eq_data : node4733 = Data.node4733 := by
  have branches : (node4721, node4783) = (Data.node4721, Data.node4783) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node4721_eq_data node4783_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4733 row children.1 children.2) row4733_eq_data branches
set_option maxRecDepth 4096 in
theorem row4799_eq_data : row4799 = Data.row4799 := rfl
def node4799 : Graph := .node 4799 row4799 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node4799_covers : Budget.Covers PrimePairVertices.largeTree 26034 node4799 :=
  Budget.node_covers _ _ 4799 row4799 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row4799_covers
theorem node4799_eq_data : node4799 = Data.node4799 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 4799 row children.1 children.2) row4799_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
