import ProjectEulerSolutions.Lib.PrimePairLarge.Graph022
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row12277_eq_data : row12277 = Data.row12277 := rfl
def node12277 : Graph := .node 12277 row12277 node12239 node12343
set_option maxRecDepth 4096 in
theorem node12277_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12277 :=
  Budget.node_covers _ _ 12277 row12277 node12239 node12343
    node12239_covers node12343_covers (by decide) (by decide) row12277_covers
theorem node12277_eq_data : node12277 = Data.node12277 := by
  have branches : (node12239, node12343) = (Data.node12239, Data.node12343) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12239_eq_data node12343_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12277 row children.1 children.2) row12277_eq_data branches
set_option maxRecDepth 4096 in
theorem row12409_eq_data : row12409 = Data.row12409 := rfl
def node12409 : Graph := .node 12409 row12409 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12409_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12409 :=
  Budget.node_covers _ _ 12409 row12409 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12409_covers
theorem node12409_eq_data : node12409 = Data.node12409 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12409 row children.1 children.2) row12409_eq_data branches
set_option maxRecDepth 4096 in
theorem row12413_eq_data : row12413 = Data.row12413 := rfl
def node12413 : Graph := .node 12413 row12413 node12409 Graph.empty
set_option maxRecDepth 4096 in
theorem node12413_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12413 :=
  Budget.node_covers _ _ 12413 row12413 node12409 Graph.empty
    node12409_covers empty_covers (by decide) (by decide) row12413_covers
theorem node12413_eq_data : node12413 = Data.node12413 := by
  have branches : (node12409, Graph.empty) = (Data.node12409, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12409_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12413 row children.1 children.2) row12413_eq_data branches
set_option maxRecDepth 4096 in
theorem row12433_eq_data : row12433 = Data.row12433 := rfl
def node12433 : Graph := .node 12433 row12433 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12433_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12433 :=
  Budget.node_covers _ _ 12433 row12433 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12433_covers
theorem node12433_eq_data : node12433 = Data.node12433 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12433 row children.1 children.2) row12433_eq_data branches
set_option maxRecDepth 4096 in
theorem row12437_eq_data : row12437 = Data.row12437 := rfl
def node12437 : Graph := .node 12437 row12437 node12433 Graph.empty
set_option maxRecDepth 4096 in
theorem node12437_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12437 :=
  Budget.node_covers _ _ 12437 row12437 node12433 Graph.empty
    node12433_covers empty_covers (by decide) (by decide) row12437_covers
theorem node12437_eq_data : node12437 = Data.node12437 := by
  have branches : (node12433, Graph.empty) = (Data.node12433, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12433_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12437 row children.1 children.2) row12437_eq_data branches
set_option maxRecDepth 4096 in
theorem row12421_eq_data : row12421 = Data.row12421 := rfl
def node12421 : Graph := .node 12421 row12421 node12413 node12437
set_option maxRecDepth 4096 in
theorem node12421_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12421 :=
  Budget.node_covers _ _ 12421 row12421 node12413 node12437
    node12413_covers node12437_covers (by decide) (by decide) row12421_covers
theorem node12421_eq_data : node12421 = Data.node12421 := by
  have branches : (node12413, node12437) = (Data.node12413, Data.node12437) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12413_eq_data node12437_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12421 row children.1 children.2) row12421_eq_data branches
set_option maxRecDepth 4096 in
theorem row12457_eq_data : row12457 = Data.row12457 := rfl
def node12457 : Graph := .node 12457 row12457 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12457_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12457 :=
  Budget.node_covers _ _ 12457 row12457 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12457_covers
theorem node12457_eq_data : node12457 = Data.node12457 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12457 row children.1 children.2) row12457_eq_data branches
set_option maxRecDepth 4096 in
theorem row12473_eq_data : row12473 = Data.row12473 := rfl
def node12473 : Graph := .node 12473 row12473 node12457 Graph.empty
set_option maxRecDepth 4096 in
theorem node12473_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12473 :=
  Budget.node_covers _ _ 12473 row12473 node12457 Graph.empty
    node12457_covers empty_covers (by decide) (by decide) row12473_covers
theorem node12473_eq_data : node12473 = Data.node12473 := by
  have branches : (node12457, Graph.empty) = (Data.node12457, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12457_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12473 row children.1 children.2) row12473_eq_data branches
set_option maxRecDepth 4096 in
theorem row12487_eq_data : row12487 = Data.row12487 := rfl
def node12487 : Graph := .node 12487 row12487 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12487_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12487 :=
  Budget.node_covers _ _ 12487 row12487 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12487_covers
theorem node12487_eq_data : node12487 = Data.node12487 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12487 row children.1 children.2) row12487_eq_data branches
set_option maxRecDepth 4096 in
theorem row12491_eq_data : row12491 = Data.row12491 := rfl
def node12491 : Graph := .node 12491 row12491 node12487 Graph.empty
set_option maxRecDepth 4096 in
theorem node12491_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12491 :=
  Budget.node_covers _ _ 12491 row12491 node12487 Graph.empty
    node12487_covers empty_covers (by decide) (by decide) row12491_covers
theorem node12491_eq_data : node12491 = Data.node12491 := by
  have branches : (node12487, Graph.empty) = (Data.node12487, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12487_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12491 row children.1 children.2) row12491_eq_data branches
set_option maxRecDepth 4096 in
theorem row12479_eq_data : row12479 = Data.row12479 := rfl
def node12479 : Graph := .node 12479 row12479 node12473 node12491
set_option maxRecDepth 4096 in
theorem node12479_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12479 :=
  Budget.node_covers _ _ 12479 row12479 node12473 node12491
    node12473_covers node12491_covers (by decide) (by decide) row12479_covers
theorem node12479_eq_data : node12479 = Data.node12479 := by
  have branches : (node12473, node12491) = (Data.node12473, Data.node12491) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12473_eq_data node12491_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12479 row children.1 children.2) row12479_eq_data branches
set_option maxRecDepth 4096 in
theorem row12451_eq_data : row12451 = Data.row12451 := rfl
def node12451 : Graph := .node 12451 row12451 node12421 node12479
set_option maxRecDepth 4096 in
theorem node12451_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12451 :=
  Budget.node_covers _ _ 12451 row12451 node12421 node12479
    node12421_covers node12479_covers (by decide) (by decide) row12451_covers
theorem node12451_eq_data : node12451 = Data.node12451 := by
  have branches : (node12421, node12479) = (Data.node12421, Data.node12479) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12421_eq_data node12479_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12451 row children.1 children.2) row12451_eq_data branches
set_option maxRecDepth 4096 in
theorem row12503_eq_data : row12503 = Data.row12503 := rfl
def node12503 : Graph := .node 12503 row12503 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12503_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12503 :=
  Budget.node_covers _ _ 12503 row12503 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12503_covers
theorem node12503_eq_data : node12503 = Data.node12503 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12503 row children.1 children.2) row12503_eq_data branches
set_option maxRecDepth 4096 in
theorem row12511_eq_data : row12511 = Data.row12511 := rfl
def node12511 : Graph := .node 12511 row12511 node12503 Graph.empty
set_option maxRecDepth 4096 in
theorem node12511_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12511 :=
  Budget.node_covers _ _ 12511 row12511 node12503 Graph.empty
    node12503_covers empty_covers (by decide) (by decide) row12511_covers
theorem node12511_eq_data : node12511 = Data.node12511 := by
  have branches : (node12503, Graph.empty) = (Data.node12503, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12503_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12511 row children.1 children.2) row12511_eq_data branches
set_option maxRecDepth 4096 in
theorem row12527_eq_data : row12527 = Data.row12527 := rfl
def node12527 : Graph := .node 12527 row12527 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12527_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12527 :=
  Budget.node_covers _ _ 12527 row12527 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12527_covers
theorem node12527_eq_data : node12527 = Data.node12527 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12527 row children.1 children.2) row12527_eq_data branches
set_option maxRecDepth 4096 in
theorem row12539_eq_data : row12539 = Data.row12539 := rfl
def node12539 : Graph := .node 12539 row12539 node12527 Graph.empty
set_option maxRecDepth 4096 in
theorem node12539_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12539 :=
  Budget.node_covers _ _ 12539 row12539 node12527 Graph.empty
    node12527_covers empty_covers (by decide) (by decide) row12539_covers
theorem node12539_eq_data : node12539 = Data.node12539 := by
  have branches : (node12527, Graph.empty) = (Data.node12527, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12527_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12539 row children.1 children.2) row12539_eq_data branches
set_option maxRecDepth 4096 in
theorem row12517_eq_data : row12517 = Data.row12517 := rfl
def node12517 : Graph := .node 12517 row12517 node12511 node12539
set_option maxRecDepth 4096 in
theorem node12517_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12517 :=
  Budget.node_covers _ _ 12517 row12517 node12511 node12539
    node12511_covers node12539_covers (by decide) (by decide) row12517_covers
theorem node12517_eq_data : node12517 = Data.node12517 := by
  have branches : (node12511, node12539) = (Data.node12511, Data.node12539) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12511_eq_data node12539_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12517 row children.1 children.2) row12517_eq_data branches
set_option maxRecDepth 4096 in
theorem row12547_eq_data : row12547 = Data.row12547 := rfl
def node12547 : Graph := .node 12547 row12547 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12547_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12547 :=
  Budget.node_covers _ _ 12547 row12547 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12547_covers
theorem node12547_eq_data : node12547 = Data.node12547 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12547 row children.1 children.2) row12547_eq_data branches
set_option maxRecDepth 4096 in
theorem row12553_eq_data : row12553 = Data.row12553 := rfl
def node12553 : Graph := .node 12553 row12553 node12547 Graph.empty
set_option maxRecDepth 4096 in
theorem node12553_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12553 :=
  Budget.node_covers _ _ 12553 row12553 node12547 Graph.empty
    node12547_covers empty_covers (by decide) (by decide) row12553_covers
theorem node12553_eq_data : node12553 = Data.node12553 := by
  have branches : (node12547, Graph.empty) = (Data.node12547, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12547_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12553 row children.1 children.2) row12553_eq_data branches
set_option maxRecDepth 4096 in
theorem row12577_eq_data : row12577 = Data.row12577 := rfl
def node12577 : Graph := .node 12577 row12577 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12577_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12577 :=
  Budget.node_covers _ _ 12577 row12577 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12577_covers
theorem node12577_eq_data : node12577 = Data.node12577 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12577 row children.1 children.2) row12577_eq_data branches
set_option maxRecDepth 4096 in
theorem row12583_eq_data : row12583 = Data.row12583 := rfl
def node12583 : Graph := .node 12583 row12583 node12577 Graph.empty
set_option maxRecDepth 4096 in
theorem node12583_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12583 :=
  Budget.node_covers _ _ 12583 row12583 node12577 Graph.empty
    node12577_covers empty_covers (by decide) (by decide) row12583_covers
theorem node12583_eq_data : node12583 = Data.node12583 := by
  have branches : (node12577, Graph.empty) = (Data.node12577, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12577_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12583 row children.1 children.2) row12583_eq_data branches
set_option maxRecDepth 4096 in
theorem row12569_eq_data : row12569 = Data.row12569 := rfl
def node12569 : Graph := .node 12569 row12569 node12553 node12583
set_option maxRecDepth 4096 in
theorem node12569_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12569 :=
  Budget.node_covers _ _ 12569 row12569 node12553 node12583
    node12553_covers node12583_covers (by decide) (by decide) row12569_covers
theorem node12569_eq_data : node12569 = Data.node12569 := by
  have branches : (node12553, node12583) = (Data.node12553, Data.node12583) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12553_eq_data node12583_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12569 row children.1 children.2) row12569_eq_data branches
set_option maxRecDepth 4096 in
theorem row12541_eq_data : row12541 = Data.row12541 := rfl
def node12541 : Graph := .node 12541 row12541 node12517 node12569
set_option maxRecDepth 4096 in
theorem node12541_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12541 :=
  Budget.node_covers _ _ 12541 row12541 node12517 node12569
    node12517_covers node12569_covers (by decide) (by decide) row12541_covers
theorem node12541_eq_data : node12541 = Data.node12541 := by
  have branches : (node12517, node12569) = (Data.node12517, Data.node12569) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12517_eq_data node12569_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12541 row children.1 children.2) row12541_eq_data branches
set_option maxRecDepth 4096 in
theorem row12497_eq_data : row12497 = Data.row12497 := rfl
def node12497 : Graph := .node 12497 row12497 node12451 node12541
set_option maxRecDepth 4096 in
theorem node12497_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12497 :=
  Budget.node_covers _ _ 12497 row12497 node12451 node12541
    node12451_covers node12541_covers (by decide) (by decide) row12497_covers
theorem node12497_eq_data : node12497 = Data.node12497 := by
  have branches : (node12451, node12541) = (Data.node12451, Data.node12541) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12451_eq_data node12541_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12497 row children.1 children.2) row12497_eq_data branches
set_option maxRecDepth 4096 in
theorem row12401_eq_data : row12401 = Data.row12401 := rfl
def node12401 : Graph := .node 12401 row12401 node12277 node12497
set_option maxRecDepth 4096 in
theorem node12401_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12401 :=
  Budget.node_covers _ _ 12401 row12401 node12277 node12497
    node12277_covers node12497_covers (by decide) (by decide) row12401_covers
theorem node12401_eq_data : node12401 = Data.node12401 := by
  have branches : (node12277, node12497) = (Data.node12277, Data.node12497) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12277_eq_data node12497_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12401 row children.1 children.2) row12401_eq_data branches
set_option maxRecDepth 4096 in
theorem row12601_eq_data : row12601 = Data.row12601 := rfl
def node12601 : Graph := .node 12601 row12601 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12601_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12601 :=
  Budget.node_covers _ _ 12601 row12601 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12601_covers
theorem node12601_eq_data : node12601 = Data.node12601 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12601 row children.1 children.2) row12601_eq_data branches
set_option maxRecDepth 4096 in
theorem row12611_eq_data : row12611 = Data.row12611 := rfl
def node12611 : Graph := .node 12611 row12611 node12601 Graph.empty
set_option maxRecDepth 4096 in
theorem node12611_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12611 :=
  Budget.node_covers _ _ 12611 row12611 node12601 Graph.empty
    node12601_covers empty_covers (by decide) (by decide) row12611_covers
theorem node12611_eq_data : node12611 = Data.node12611 := by
  have branches : (node12601, Graph.empty) = (Data.node12601, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12601_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12611 row children.1 children.2) row12611_eq_data branches
set_option maxRecDepth 4096 in
theorem row12619_eq_data : row12619 = Data.row12619 := rfl
def node12619 : Graph := .node 12619 row12619 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12619_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12619 :=
  Budget.node_covers _ _ 12619 row12619 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12619_covers
theorem node12619_eq_data : node12619 = Data.node12619 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12619 row children.1 children.2) row12619_eq_data branches
set_option maxRecDepth 4096 in
theorem row12637_eq_data : row12637 = Data.row12637 := rfl
def node12637 : Graph := .node 12637 row12637 node12619 Graph.empty
set_option maxRecDepth 4096 in
theorem node12637_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12637 :=
  Budget.node_covers _ _ 12637 row12637 node12619 Graph.empty
    node12619_covers empty_covers (by decide) (by decide) row12637_covers
theorem node12637_eq_data : node12637 = Data.node12637 := by
  have branches : (node12619, Graph.empty) = (Data.node12619, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12619_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12637 row children.1 children.2) row12637_eq_data branches
set_option maxRecDepth 4096 in
theorem row12613_eq_data : row12613 = Data.row12613 := rfl
def node12613 : Graph := .node 12613 row12613 node12611 node12637
set_option maxRecDepth 4096 in
theorem node12613_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12613 :=
  Budget.node_covers _ _ 12613 row12613 node12611 node12637
    node12611_covers node12637_covers (by decide) (by decide) row12613_covers
theorem node12613_eq_data : node12613 = Data.node12613 := by
  have branches : (node12611, node12637) = (Data.node12611, Data.node12637) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12611_eq_data node12637_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12613 row children.1 children.2) row12613_eq_data branches
set_option maxRecDepth 4096 in
theorem row12647_eq_data : row12647 = Data.row12647 := rfl
def node12647 : Graph := .node 12647 row12647 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12647_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12647 :=
  Budget.node_covers _ _ 12647 row12647 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12647_covers
theorem node12647_eq_data : node12647 = Data.node12647 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12647 row children.1 children.2) row12647_eq_data branches
set_option maxRecDepth 4096 in
theorem row12653_eq_data : row12653 = Data.row12653 := rfl
def node12653 : Graph := .node 12653 row12653 node12647 Graph.empty
set_option maxRecDepth 4096 in
theorem node12653_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12653 :=
  Budget.node_covers _ _ 12653 row12653 node12647 Graph.empty
    node12647_covers empty_covers (by decide) (by decide) row12653_covers
theorem node12653_eq_data : node12653 = Data.node12653 := by
  have branches : (node12647, Graph.empty) = (Data.node12647, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12647_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12653 row children.1 children.2) row12653_eq_data branches
set_option maxRecDepth 4096 in
theorem row12671_eq_data : row12671 = Data.row12671 := rfl
def node12671 : Graph := .node 12671 row12671 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12671_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12671 :=
  Budget.node_covers _ _ 12671 row12671 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12671_covers
theorem node12671_eq_data : node12671 = Data.node12671 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12671 row children.1 children.2) row12671_eq_data branches
set_option maxRecDepth 4096 in
theorem row12689_eq_data : row12689 = Data.row12689 := rfl
def node12689 : Graph := .node 12689 row12689 node12671 Graph.empty
set_option maxRecDepth 4096 in
theorem node12689_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12689 :=
  Budget.node_covers _ _ 12689 row12689 node12671 Graph.empty
    node12671_covers empty_covers (by decide) (by decide) row12689_covers
theorem node12689_eq_data : node12689 = Data.node12689 := by
  have branches : (node12671, Graph.empty) = (Data.node12671, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12671_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12689 row children.1 children.2) row12689_eq_data branches
set_option maxRecDepth 4096 in
theorem row12659_eq_data : row12659 = Data.row12659 := rfl
def node12659 : Graph := .node 12659 row12659 node12653 node12689
set_option maxRecDepth 4096 in
theorem node12659_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12659 :=
  Budget.node_covers _ _ 12659 row12659 node12653 node12689
    node12653_covers node12689_covers (by decide) (by decide) row12659_covers
theorem node12659_eq_data : node12659 = Data.node12659 := by
  have branches : (node12653, node12689) = (Data.node12653, Data.node12689) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12653_eq_data node12689_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12659 row children.1 children.2) row12659_eq_data branches
set_option maxRecDepth 4096 in
theorem row12641_eq_data : row12641 = Data.row12641 := rfl
def node12641 : Graph := .node 12641 row12641 node12613 node12659
set_option maxRecDepth 4096 in
theorem node12641_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12641 :=
  Budget.node_covers _ _ 12641 row12641 node12613 node12659
    node12613_covers node12659_covers (by decide) (by decide) row12641_covers
theorem node12641_eq_data : node12641 = Data.node12641 := by
  have branches : (node12613, node12659) = (Data.node12613, Data.node12659) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12613_eq_data node12659_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12641 row children.1 children.2) row12641_eq_data branches
set_option maxRecDepth 4096 in
theorem row12703_eq_data : row12703 = Data.row12703 := rfl
def node12703 : Graph := .node 12703 row12703 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12703_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12703 :=
  Budget.node_covers _ _ 12703 row12703 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12703_covers
theorem node12703_eq_data : node12703 = Data.node12703 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12703 row children.1 children.2) row12703_eq_data branches
set_option maxRecDepth 4096 in
theorem row12713_eq_data : row12713 = Data.row12713 := rfl
def node12713 : Graph := .node 12713 row12713 node12703 Graph.empty
set_option maxRecDepth 4096 in
theorem node12713_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12713 :=
  Budget.node_covers _ _ 12713 row12713 node12703 Graph.empty
    node12703_covers empty_covers (by decide) (by decide) row12713_covers
theorem node12713_eq_data : node12713 = Data.node12713 := by
  have branches : (node12703, Graph.empty) = (Data.node12703, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12703_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12713 row children.1 children.2) row12713_eq_data branches
set_option maxRecDepth 4096 in
theorem row12739_eq_data : row12739 = Data.row12739 := rfl
def node12739 : Graph := .node 12739 row12739 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12739_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12739 :=
  Budget.node_covers _ _ 12739 row12739 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12739_covers
theorem node12739_eq_data : node12739 = Data.node12739 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12739 row children.1 children.2) row12739_eq_data branches
set_option maxRecDepth 4096 in
theorem row12743_eq_data : row12743 = Data.row12743 := rfl
def node12743 : Graph := .node 12743 row12743 node12739 Graph.empty
set_option maxRecDepth 4096 in
theorem node12743_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12743 :=
  Budget.node_covers _ _ 12743 row12743 node12739 Graph.empty
    node12739_covers empty_covers (by decide) (by decide) row12743_covers
theorem node12743_eq_data : node12743 = Data.node12743 := by
  have branches : (node12739, Graph.empty) = (Data.node12739, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12739_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12743 row children.1 children.2) row12743_eq_data branches
set_option maxRecDepth 4096 in
theorem row12721_eq_data : row12721 = Data.row12721 := rfl
def node12721 : Graph := .node 12721 row12721 node12713 node12743
set_option maxRecDepth 4096 in
theorem node12721_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12721 :=
  Budget.node_covers _ _ 12721 row12721 node12713 node12743
    node12713_covers node12743_covers (by decide) (by decide) row12721_covers
theorem node12721_eq_data : node12721 = Data.node12721 := by
  have branches : (node12713, node12743) = (Data.node12713, Data.node12743) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12713_eq_data node12743_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12721 row children.1 children.2) row12721_eq_data branches
set_option maxRecDepth 4096 in
theorem row12763_eq_data : row12763 = Data.row12763 := rfl
def node12763 : Graph := .node 12763 row12763 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12763_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12763 :=
  Budget.node_covers _ _ 12763 row12763 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12763_covers
theorem node12763_eq_data : node12763 = Data.node12763 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12763 row children.1 children.2) row12763_eq_data branches
set_option maxRecDepth 4096 in
theorem row12781_eq_data : row12781 = Data.row12781 := rfl
def node12781 : Graph := .node 12781 row12781 node12763 Graph.empty
set_option maxRecDepth 4096 in
theorem node12781_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12781 :=
  Budget.node_covers _ _ 12781 row12781 node12763 Graph.empty
    node12763_covers empty_covers (by decide) (by decide) row12781_covers
theorem node12781_eq_data : node12781 = Data.node12781 := by
  have branches : (node12763, Graph.empty) = (Data.node12763, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12763_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12781 row children.1 children.2) row12781_eq_data branches
set_option maxRecDepth 4096 in
theorem row12799_eq_data : row12799 = Data.row12799 := rfl
def node12799 : Graph := .node 12799 row12799 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12799_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12799 :=
  Budget.node_covers _ _ 12799 row12799 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12799_covers
theorem node12799_eq_data : node12799 = Data.node12799 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12799 row children.1 children.2) row12799_eq_data branches
set_option maxRecDepth 4096 in
theorem row12809_eq_data : row12809 = Data.row12809 := rfl
def node12809 : Graph := .node 12809 row12809 node12799 Graph.empty
set_option maxRecDepth 4096 in
theorem node12809_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12809 :=
  Budget.node_covers _ _ 12809 row12809 node12799 Graph.empty
    node12799_covers empty_covers (by decide) (by decide) row12809_covers
theorem node12809_eq_data : node12809 = Data.node12809 := by
  have branches : (node12799, Graph.empty) = (Data.node12799, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12799_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12809 row children.1 children.2) row12809_eq_data branches
set_option maxRecDepth 4096 in
theorem row12791_eq_data : row12791 = Data.row12791 := rfl
def node12791 : Graph := .node 12791 row12791 node12781 node12809
set_option maxRecDepth 4096 in
theorem node12791_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12791 :=
  Budget.node_covers _ _ 12791 row12791 node12781 node12809
    node12781_covers node12809_covers (by decide) (by decide) row12791_covers
theorem node12791_eq_data : node12791 = Data.node12791 := by
  have branches : (node12781, node12809) = (Data.node12781, Data.node12809) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12781_eq_data node12809_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12791 row children.1 children.2) row12791_eq_data branches
set_option maxRecDepth 4096 in
theorem row12757_eq_data : row12757 = Data.row12757 := rfl
def node12757 : Graph := .node 12757 row12757 node12721 node12791
set_option maxRecDepth 4096 in
theorem node12757_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12757 :=
  Budget.node_covers _ _ 12757 row12757 node12721 node12791
    node12721_covers node12791_covers (by decide) (by decide) row12757_covers
theorem node12757_eq_data : node12757 = Data.node12757 := by
  have branches : (node12721, node12791) = (Data.node12721, Data.node12791) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12721_eq_data node12791_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12757 row children.1 children.2) row12757_eq_data branches
set_option maxRecDepth 4096 in
theorem row12697_eq_data : row12697 = Data.row12697 := rfl
def node12697 : Graph := .node 12697 row12697 node12641 node12757
set_option maxRecDepth 4096 in
theorem node12697_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12697 :=
  Budget.node_covers _ _ 12697 row12697 node12641 node12757
    node12641_covers node12757_covers (by decide) (by decide) row12697_covers
theorem node12697_eq_data : node12697 = Data.node12697 := by
  have branches : (node12641, node12757) = (Data.node12641, Data.node12757) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12641_eq_data node12757_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12697 row children.1 children.2) row12697_eq_data branches
set_option maxRecDepth 4096 in
theorem row12823_eq_data : row12823 = Data.row12823 := rfl
def node12823 : Graph := .node 12823 row12823 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12823_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12823 :=
  Budget.node_covers _ _ 12823 row12823 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12823_covers
theorem node12823_eq_data : node12823 = Data.node12823 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12823 row children.1 children.2) row12823_eq_data branches
set_option maxRecDepth 4096 in
theorem row12829_eq_data : row12829 = Data.row12829 := rfl
def node12829 : Graph := .node 12829 row12829 node12823 Graph.empty
set_option maxRecDepth 4096 in
theorem node12829_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12829 :=
  Budget.node_covers _ _ 12829 row12829 node12823 Graph.empty
    node12823_covers empty_covers (by decide) (by decide) row12829_covers
theorem node12829_eq_data : node12829 = Data.node12829 := by
  have branches : (node12823, Graph.empty) = (Data.node12823, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12823_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12829 row children.1 children.2) row12829_eq_data branches
set_option maxRecDepth 4096 in
theorem row12853_eq_data : row12853 = Data.row12853 := rfl
def node12853 : Graph := .node 12853 row12853 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12853_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12853 :=
  Budget.node_covers _ _ 12853 row12853 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12853_covers
theorem node12853_eq_data : node12853 = Data.node12853 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12853 row children.1 children.2) row12853_eq_data branches
set_option maxRecDepth 4096 in
theorem row12889_eq_data : row12889 = Data.row12889 := rfl
def node12889 : Graph := .node 12889 row12889 node12853 Graph.empty
set_option maxRecDepth 4096 in
theorem node12889_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12889 :=
  Budget.node_covers _ _ 12889 row12889 node12853 Graph.empty
    node12853_covers empty_covers (by decide) (by decide) row12889_covers
theorem node12889_eq_data : node12889 = Data.node12889 := by
  have branches : (node12853, Graph.empty) = (Data.node12853, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12853_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12889 row children.1 children.2) row12889_eq_data branches
set_option maxRecDepth 4096 in
theorem row12841_eq_data : row12841 = Data.row12841 := rfl
def node12841 : Graph := .node 12841 row12841 node12829 node12889
set_option maxRecDepth 4096 in
theorem node12841_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12841 :=
  Budget.node_covers _ _ 12841 row12841 node12829 node12889
    node12829_covers node12889_covers (by decide) (by decide) row12841_covers
theorem node12841_eq_data : node12841 = Data.node12841 := by
  have branches : (node12829, node12889) = (Data.node12829, Data.node12889) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12829_eq_data node12889_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12841 row children.1 children.2) row12841_eq_data branches
set_option maxRecDepth 4096 in
theorem row12899_eq_data : row12899 = Data.row12899 := rfl
def node12899 : Graph := .node 12899 row12899 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12899_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12899 :=
  Budget.node_covers _ _ 12899 row12899 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12899_covers
theorem node12899_eq_data : node12899 = Data.node12899 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12899 row children.1 children.2) row12899_eq_data branches
set_option maxRecDepth 4096 in
theorem row12907_eq_data : row12907 = Data.row12907 := rfl
def node12907 : Graph := .node 12907 row12907 node12899 Graph.empty
set_option maxRecDepth 4096 in
theorem node12907_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12907 :=
  Budget.node_covers _ _ 12907 row12907 node12899 Graph.empty
    node12899_covers empty_covers (by decide) (by decide) row12907_covers
theorem node12907_eq_data : node12907 = Data.node12907 := by
  have branches : (node12899, Graph.empty) = (Data.node12899, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12899_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12907 row children.1 children.2) row12907_eq_data branches
set_option maxRecDepth 4096 in
theorem row12917_eq_data : row12917 = Data.row12917 := rfl
def node12917 : Graph := .node 12917 row12917 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12917_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12917 :=
  Budget.node_covers _ _ 12917 row12917 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12917_covers
theorem node12917_eq_data : node12917 = Data.node12917 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12917 row children.1 children.2) row12917_eq_data branches
set_option maxRecDepth 4096 in
theorem row12919_eq_data : row12919 = Data.row12919 := rfl
def node12919 : Graph := .node 12919 row12919 node12917 Graph.empty
set_option maxRecDepth 4096 in
theorem node12919_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12919 :=
  Budget.node_covers _ _ 12919 row12919 node12917 Graph.empty
    node12917_covers empty_covers (by decide) (by decide) row12919_covers
theorem node12919_eq_data : node12919 = Data.node12919 := by
  have branches : (node12917, Graph.empty) = (Data.node12917, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12917_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12919 row children.1 children.2) row12919_eq_data branches
set_option maxRecDepth 4096 in
theorem row12911_eq_data : row12911 = Data.row12911 := rfl
def node12911 : Graph := .node 12911 row12911 node12907 node12919
set_option maxRecDepth 4096 in
theorem node12911_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12911 :=
  Budget.node_covers _ _ 12911 row12911 node12907 node12919
    node12907_covers node12919_covers (by decide) (by decide) row12911_covers
theorem node12911_eq_data : node12911 = Data.node12911 := by
  have branches : (node12907, node12919) = (Data.node12907, Data.node12919) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12907_eq_data node12919_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12911 row children.1 children.2) row12911_eq_data branches
set_option maxRecDepth 4096 in
theorem row12893_eq_data : row12893 = Data.row12893 := rfl
def node12893 : Graph := .node 12893 row12893 node12841 node12911
set_option maxRecDepth 4096 in
theorem node12893_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12893 :=
  Budget.node_covers _ _ 12893 row12893 node12841 node12911
    node12841_covers node12911_covers (by decide) (by decide) row12893_covers
theorem node12893_eq_data : node12893 = Data.node12893 := by
  have branches : (node12841, node12911) = (Data.node12841, Data.node12911) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12841_eq_data node12911_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12893 row children.1 children.2) row12893_eq_data branches
set_option maxRecDepth 4096 in
theorem row12941_eq_data : row12941 = Data.row12941 := rfl
def node12941 : Graph := .node 12941 row12941 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12941_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12941 :=
  Budget.node_covers _ _ 12941 row12941 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12941_covers
theorem node12941_eq_data : node12941 = Data.node12941 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12941 row children.1 children.2) row12941_eq_data branches
set_option maxRecDepth 4096 in
theorem row12953_eq_data : row12953 = Data.row12953 := rfl
def node12953 : Graph := .node 12953 row12953 node12941 Graph.empty
set_option maxRecDepth 4096 in
theorem node12953_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12953 :=
  Budget.node_covers _ _ 12953 row12953 node12941 Graph.empty
    node12941_covers empty_covers (by decide) (by decide) row12953_covers
theorem node12953_eq_data : node12953 = Data.node12953 := by
  have branches : (node12941, Graph.empty) = (Data.node12941, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12941_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12953 row children.1 children.2) row12953_eq_data branches
set_option maxRecDepth 4096 in
theorem row12967_eq_data : row12967 = Data.row12967 := rfl
def node12967 : Graph := .node 12967 row12967 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node12967_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12967 :=
  Budget.node_covers _ _ 12967 row12967 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row12967_covers
theorem node12967_eq_data : node12967 = Data.node12967 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12967 row children.1 children.2) row12967_eq_data branches
set_option maxRecDepth 4096 in
theorem row12973_eq_data : row12973 = Data.row12973 := rfl
def node12973 : Graph := .node 12973 row12973 node12967 Graph.empty
set_option maxRecDepth 4096 in
theorem node12973_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12973 :=
  Budget.node_covers _ _ 12973 row12973 node12967 Graph.empty
    node12967_covers empty_covers (by decide) (by decide) row12973_covers
theorem node12973_eq_data : node12973 = Data.node12973 := by
  have branches : (node12967, Graph.empty) = (Data.node12967, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12967_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12973 row children.1 children.2) row12973_eq_data branches
set_option maxRecDepth 4096 in
theorem row12959_eq_data : row12959 = Data.row12959 := rfl
def node12959 : Graph := .node 12959 row12959 node12953 node12973
set_option maxRecDepth 4096 in
theorem node12959_covers : Budget.Covers PrimePairVertices.largeTree 26034 node12959 :=
  Budget.node_covers _ _ 12959 row12959 node12953 node12973
    node12953_covers node12973_covers (by decide) (by decide) row12959_covers
theorem node12959_eq_data : node12959 = Data.node12959 := by
  have branches : (node12953, node12973) = (Data.node12953, Data.node12973) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node12953_eq_data node12973_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 12959 row children.1 children.2) row12959_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
