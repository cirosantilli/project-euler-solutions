import ProjectEulerSolutions.Lib.PrimePairLarge.Graph017
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row9403_eq_data : row9403 = Data.row9403 := rfl
def node9403 : Graph := .node 9403 row9403 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9403_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9403 :=
  Budget.node_covers _ _ 9403 row9403 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9403_covers
theorem node9403_eq_data : node9403 = Data.node9403 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9403 row children.1 children.2) row9403_eq_data branches
set_option maxRecDepth 4096 in
theorem row9413_eq_data : row9413 = Data.row9413 := rfl
def node9413 : Graph := .node 9413 row9413 node9403 Graph.empty
set_option maxRecDepth 4096 in
theorem node9413_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9413 :=
  Budget.node_covers _ _ 9413 row9413 node9403 Graph.empty
    node9403_covers empty_covers (by decide) (by decide) row9413_covers
theorem node9413_eq_data : node9413 = Data.node9413 := by
  have branches : (node9403, Graph.empty) = (Data.node9403, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9403_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9413 row children.1 children.2) row9413_eq_data branches
set_option maxRecDepth 4096 in
theorem row9397_eq_data : row9397 = Data.row9397 := rfl
def node9397 : Graph := .node 9397 row9397 node9391 node9413
set_option maxRecDepth 4096 in
theorem node9397_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9397 :=
  Budget.node_covers _ _ 9397 row9397 node9391 node9413
    node9391_covers node9413_covers (by decide) (by decide) row9397_covers
theorem node9397_eq_data : node9397 = Data.node9397 := by
  have branches : (node9391, node9413) = (Data.node9391, Data.node9413) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9391_eq_data node9413_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9397 row children.1 children.2) row9397_eq_data branches
set_option maxRecDepth 4096 in
theorem row9371_eq_data : row9371 = Data.row9371 := rfl
def node9371 : Graph := .node 9371 row9371 node9341 node9397
set_option maxRecDepth 4096 in
theorem node9371_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9371 :=
  Budget.node_covers _ _ 9371 row9371 node9341 node9397
    node9341_covers node9397_covers (by decide) (by decide) row9371_covers
theorem node9371_eq_data : node9371 = Data.node9371 := by
  have branches : (node9341, node9397) = (Data.node9341, Data.node9397) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9341_eq_data node9397_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9371 row children.1 children.2) row9371_eq_data branches
set_option maxRecDepth 4096 in
theorem row9319_eq_data : row9319 = Data.row9319 := rfl
def node9319 : Graph := .node 9319 row9319 node9257 node9371
set_option maxRecDepth 4096 in
theorem node9319_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9319 :=
  Budget.node_covers _ _ 9319 row9319 node9257 node9371
    node9257_covers node9371_covers (by decide) (by decide) row9319_covers
theorem node9319_eq_data : node9319 = Data.node9319 := by
  have branches : (node9257, node9371) = (Data.node9257, Data.node9371) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9257_eq_data node9371_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9319 row children.1 children.2) row9319_eq_data branches
set_option maxRecDepth 4096 in
theorem row9203_eq_data : row9203 = Data.row9203 := rfl
def node9203 : Graph := .node 9203 row9203 node9103 node9319
set_option maxRecDepth 4096 in
theorem node9203_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9203 :=
  Budget.node_covers _ _ 9203 row9203 node9103 node9319
    node9103_covers node9319_covers (by decide) (by decide) row9203_covers
theorem node9203_eq_data : node9203 = Data.node9203 := by
  have branches : (node9103, node9319) = (Data.node9103, Data.node9319) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9103_eq_data node9319_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9203 row children.1 children.2) row9203_eq_data branches
set_option maxRecDepth 4096 in
theorem row8999_eq_data : row8999 = Data.row8999 := rfl
def node8999 : Graph := .node 8999 row8999 node8761 node9203
set_option maxRecDepth 4096 in
theorem node8999_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8999 :=
  Budget.node_covers _ _ 8999 row8999 node8761 node9203
    node8761_covers node9203_covers (by decide) (by decide) row8999_covers
theorem node8999_eq_data : node8999 = Data.node8999 := by
  have branches : (node8761, node9203) = (Data.node8761, Data.node9203) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8761_eq_data node9203_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8999 row children.1 children.2) row8999_eq_data branches
set_option maxRecDepth 4096 in
theorem row8573_eq_data : row8573 = Data.row8573 := rfl
def node8573 : Graph := .node 8573 row8573 node8111 node8999
set_option maxRecDepth 4096 in
theorem node8573_covers : Budget.Covers PrimePairVertices.largeTree 26034 node8573 :=
  Budget.node_covers _ _ 8573 row8573 node8111 node8999
    node8111_covers node8999_covers (by decide) (by decide) row8573_covers
theorem node8573_eq_data : node8573 = Data.node8573 := by
  have branches : (node8111, node8999) = (Data.node8111, Data.node8999) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node8111_eq_data node8999_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 8573 row children.1 children.2) row8573_eq_data branches
set_option maxRecDepth 4096 in
theorem row7649_eq_data : row7649 = Data.row7649 := rfl
def node7649 : Graph := .node 7649 row7649 node6791 node8573
set_option maxRecDepth 4096 in
theorem node7649_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7649 :=
  Budget.node_covers _ _ 7649 row7649 node6791 node8573
    node6791_covers node8573_covers (by decide) (by decide) row7649_covers
theorem node7649_eq_data : node7649 = Data.node7649 := by
  have branches : (node6791, node8573) = (Data.node6791, Data.node8573) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6791_eq_data node8573_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7649 row children.1 children.2) row7649_eq_data branches
set_option maxRecDepth 4096 in
theorem row9421_eq_data : row9421 = Data.row9421 := rfl
def node9421 : Graph := .node 9421 row9421 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9421_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9421 :=
  Budget.node_covers _ _ 9421 row9421 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9421_covers
theorem node9421_eq_data : node9421 = Data.node9421 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9421 row children.1 children.2) row9421_eq_data branches
set_option maxRecDepth 4096 in
theorem row9433_eq_data : row9433 = Data.row9433 := rfl
def node9433 : Graph := .node 9433 row9433 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9433_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9433 :=
  Budget.node_covers _ _ 9433 row9433 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9433_covers
theorem node9433_eq_data : node9433 = Data.node9433 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9433 row children.1 children.2) row9433_eq_data branches
set_option maxRecDepth 4096 in
theorem row9431_eq_data : row9431 = Data.row9431 := rfl
def node9431 : Graph := .node 9431 row9431 node9421 node9433
set_option maxRecDepth 4096 in
theorem node9431_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9431 :=
  Budget.node_covers _ _ 9431 row9431 node9421 node9433
    node9421_covers node9433_covers (by decide) (by decide) row9431_covers
theorem node9431_eq_data : node9431 = Data.node9431 := by
  have branches : (node9421, node9433) = (Data.node9421, Data.node9433) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9421_eq_data node9433_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9431 row children.1 children.2) row9431_eq_data branches
set_option maxRecDepth 4096 in
theorem row9439_eq_data : row9439 = Data.row9439 := rfl
def node9439 : Graph := .node 9439 row9439 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9439_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9439 :=
  Budget.node_covers _ _ 9439 row9439 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9439_covers
theorem node9439_eq_data : node9439 = Data.node9439 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9439 row children.1 children.2) row9439_eq_data branches
set_option maxRecDepth 4096 in
theorem row9461_eq_data : row9461 = Data.row9461 := rfl
def node9461 : Graph := .node 9461 row9461 node9439 Graph.empty
set_option maxRecDepth 4096 in
theorem node9461_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9461 :=
  Budget.node_covers _ _ 9461 row9461 node9439 Graph.empty
    node9439_covers empty_covers (by decide) (by decide) row9461_covers
theorem node9461_eq_data : node9461 = Data.node9461 := by
  have branches : (node9439, Graph.empty) = (Data.node9439, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9439_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9461 row children.1 children.2) row9461_eq_data branches
set_option maxRecDepth 4096 in
theorem row9437_eq_data : row9437 = Data.row9437 := rfl
def node9437 : Graph := .node 9437 row9437 node9431 node9461
set_option maxRecDepth 4096 in
theorem node9437_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9437 :=
  Budget.node_covers _ _ 9437 row9437 node9431 node9461
    node9431_covers node9461_covers (by decide) (by decide) row9437_covers
theorem node9437_eq_data : node9437 = Data.node9437 := by
  have branches : (node9431, node9461) = (Data.node9431, Data.node9461) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9431_eq_data node9461_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9437 row children.1 children.2) row9437_eq_data branches
set_option maxRecDepth 4096 in
theorem row9467_eq_data : row9467 = Data.row9467 := rfl
def node9467 : Graph := .node 9467 row9467 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9467_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9467 :=
  Budget.node_covers _ _ 9467 row9467 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9467_covers
theorem node9467_eq_data : node9467 = Data.node9467 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9467 row children.1 children.2) row9467_eq_data branches
set_option maxRecDepth 4096 in
theorem row9473_eq_data : row9473 = Data.row9473 := rfl
def node9473 : Graph := .node 9473 row9473 node9467 Graph.empty
set_option maxRecDepth 4096 in
theorem node9473_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9473 :=
  Budget.node_covers _ _ 9473 row9473 node9467 Graph.empty
    node9467_covers empty_covers (by decide) (by decide) row9473_covers
theorem node9473_eq_data : node9473 = Data.node9473 := by
  have branches : (node9467, Graph.empty) = (Data.node9467, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9467_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9473 row children.1 children.2) row9473_eq_data branches
set_option maxRecDepth 4096 in
theorem row9491_eq_data : row9491 = Data.row9491 := rfl
def node9491 : Graph := .node 9491 row9491 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9491_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9491 :=
  Budget.node_covers _ _ 9491 row9491 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9491_covers
theorem node9491_eq_data : node9491 = Data.node9491 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9491 row children.1 children.2) row9491_eq_data branches
set_option maxRecDepth 4096 in
theorem row9497_eq_data : row9497 = Data.row9497 := rfl
def node9497 : Graph := .node 9497 row9497 node9491 Graph.empty
set_option maxRecDepth 4096 in
theorem node9497_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9497 :=
  Budget.node_covers _ _ 9497 row9497 node9491 Graph.empty
    node9491_covers empty_covers (by decide) (by decide) row9497_covers
theorem node9497_eq_data : node9497 = Data.node9497 := by
  have branches : (node9491, Graph.empty) = (Data.node9491, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9491_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9497 row children.1 children.2) row9497_eq_data branches
set_option maxRecDepth 4096 in
theorem row9479_eq_data : row9479 = Data.row9479 := rfl
def node9479 : Graph := .node 9479 row9479 node9473 node9497
set_option maxRecDepth 4096 in
theorem node9479_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9479 :=
  Budget.node_covers _ _ 9479 row9479 node9473 node9497
    node9473_covers node9497_covers (by decide) (by decide) row9479_covers
theorem node9479_eq_data : node9479 = Data.node9479 := by
  have branches : (node9473, node9497) = (Data.node9473, Data.node9497) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9473_eq_data node9497_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9479 row children.1 children.2) row9479_eq_data branches
set_option maxRecDepth 4096 in
theorem row9463_eq_data : row9463 = Data.row9463 := rfl
def node9463 : Graph := .node 9463 row9463 node9437 node9479
set_option maxRecDepth 4096 in
theorem node9463_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9463 :=
  Budget.node_covers _ _ 9463 row9463 node9437 node9479
    node9437_covers node9479_covers (by decide) (by decide) row9463_covers
theorem node9463_eq_data : node9463 = Data.node9463 := by
  have branches : (node9437, node9479) = (Data.node9437, Data.node9479) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9437_eq_data node9479_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9463 row children.1 children.2) row9463_eq_data branches
set_option maxRecDepth 4096 in
theorem row9521_eq_data : row9521 = Data.row9521 := rfl
def node9521 : Graph := .node 9521 row9521 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9521_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9521 :=
  Budget.node_covers _ _ 9521 row9521 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9521_covers
theorem node9521_eq_data : node9521 = Data.node9521 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9521 row children.1 children.2) row9521_eq_data branches
set_option maxRecDepth 4096 in
theorem row9533_eq_data : row9533 = Data.row9533 := rfl
def node9533 : Graph := .node 9533 row9533 node9521 Graph.empty
set_option maxRecDepth 4096 in
theorem node9533_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9533 :=
  Budget.node_covers _ _ 9533 row9533 node9521 Graph.empty
    node9521_covers empty_covers (by decide) (by decide) row9533_covers
theorem node9533_eq_data : node9533 = Data.node9533 := by
  have branches : (node9521, Graph.empty) = (Data.node9521, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9521_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9533 row children.1 children.2) row9533_eq_data branches
set_option maxRecDepth 4096 in
theorem row9547_eq_data : row9547 = Data.row9547 := rfl
def node9547 : Graph := .node 9547 row9547 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9547_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9547 :=
  Budget.node_covers _ _ 9547 row9547 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9547_covers
theorem node9547_eq_data : node9547 = Data.node9547 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9547 row children.1 children.2) row9547_eq_data branches
set_option maxRecDepth 4096 in
theorem row9551_eq_data : row9551 = Data.row9551 := rfl
def node9551 : Graph := .node 9551 row9551 node9547 Graph.empty
set_option maxRecDepth 4096 in
theorem node9551_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9551 :=
  Budget.node_covers _ _ 9551 row9551 node9547 Graph.empty
    node9547_covers empty_covers (by decide) (by decide) row9551_covers
theorem node9551_eq_data : node9551 = Data.node9551 := by
  have branches : (node9547, Graph.empty) = (Data.node9547, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9547_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9551 row children.1 children.2) row9551_eq_data branches
set_option maxRecDepth 4096 in
theorem row9539_eq_data : row9539 = Data.row9539 := rfl
def node9539 : Graph := .node 9539 row9539 node9533 node9551
set_option maxRecDepth 4096 in
theorem node9539_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9539 :=
  Budget.node_covers _ _ 9539 row9539 node9533 node9551
    node9533_covers node9551_covers (by decide) (by decide) row9539_covers
theorem node9539_eq_data : node9539 = Data.node9539 := by
  have branches : (node9533, node9551) = (Data.node9533, Data.node9551) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9533_eq_data node9551_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9539 row children.1 children.2) row9539_eq_data branches
set_option maxRecDepth 4096 in
theorem row9601_eq_data : row9601 = Data.row9601 := rfl
def node9601 : Graph := .node 9601 row9601 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9601_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9601 :=
  Budget.node_covers _ _ 9601 row9601 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9601_covers
theorem node9601_eq_data : node9601 = Data.node9601 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9601 row children.1 children.2) row9601_eq_data branches
set_option maxRecDepth 4096 in
theorem row9613_eq_data : row9613 = Data.row9613 := rfl
def node9613 : Graph := .node 9613 row9613 node9601 Graph.empty
set_option maxRecDepth 4096 in
theorem node9613_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9613 :=
  Budget.node_covers _ _ 9613 row9613 node9601 Graph.empty
    node9601_covers empty_covers (by decide) (by decide) row9613_covers
theorem node9613_eq_data : node9613 = Data.node9613 := by
  have branches : (node9601, Graph.empty) = (Data.node9601, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9601_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9613 row children.1 children.2) row9613_eq_data branches
set_option maxRecDepth 4096 in
theorem row9623_eq_data : row9623 = Data.row9623 := rfl
def node9623 : Graph := .node 9623 row9623 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9623_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9623 :=
  Budget.node_covers _ _ 9623 row9623 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9623_covers
theorem node9623_eq_data : node9623 = Data.node9623 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9623 row children.1 children.2) row9623_eq_data branches
set_option maxRecDepth 4096 in
theorem row9629_eq_data : row9629 = Data.row9629 := rfl
def node9629 : Graph := .node 9629 row9629 node9623 Graph.empty
set_option maxRecDepth 4096 in
theorem node9629_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9629 :=
  Budget.node_covers _ _ 9629 row9629 node9623 Graph.empty
    node9623_covers empty_covers (by decide) (by decide) row9629_covers
theorem node9629_eq_data : node9629 = Data.node9629 := by
  have branches : (node9623, Graph.empty) = (Data.node9623, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9623_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9629 row children.1 children.2) row9629_eq_data branches
set_option maxRecDepth 4096 in
theorem row9619_eq_data : row9619 = Data.row9619 := rfl
def node9619 : Graph := .node 9619 row9619 node9613 node9629
set_option maxRecDepth 4096 in
theorem node9619_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9619 :=
  Budget.node_covers _ _ 9619 row9619 node9613 node9629
    node9613_covers node9629_covers (by decide) (by decide) row9619_covers
theorem node9619_eq_data : node9619 = Data.node9619 := by
  have branches : (node9613, node9629) = (Data.node9613, Data.node9629) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9613_eq_data node9629_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9619 row children.1 children.2) row9619_eq_data branches
set_option maxRecDepth 4096 in
theorem row9587_eq_data : row9587 = Data.row9587 := rfl
def node9587 : Graph := .node 9587 row9587 node9539 node9619
set_option maxRecDepth 4096 in
theorem node9587_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9587 :=
  Budget.node_covers _ _ 9587 row9587 node9539 node9619
    node9539_covers node9619_covers (by decide) (by decide) row9587_covers
theorem node9587_eq_data : node9587 = Data.node9587 := by
  have branches : (node9539, node9619) = (Data.node9539, Data.node9619) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9539_eq_data node9619_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9587 row children.1 children.2) row9587_eq_data branches
set_option maxRecDepth 4096 in
theorem row9511_eq_data : row9511 = Data.row9511 := rfl
def node9511 : Graph := .node 9511 row9511 node9463 node9587
set_option maxRecDepth 4096 in
theorem node9511_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9511 :=
  Budget.node_covers _ _ 9511 row9511 node9463 node9587
    node9463_covers node9587_covers (by decide) (by decide) row9511_covers
theorem node9511_eq_data : node9511 = Data.node9511 := by
  have branches : (node9463, node9587) = (Data.node9463, Data.node9587) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9463_eq_data node9587_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9511 row children.1 children.2) row9511_eq_data branches
set_option maxRecDepth 4096 in
theorem row9643_eq_data : row9643 = Data.row9643 := rfl
def node9643 : Graph := .node 9643 row9643 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9643_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9643 :=
  Budget.node_covers _ _ 9643 row9643 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9643_covers
theorem node9643_eq_data : node9643 = Data.node9643 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9643 row children.1 children.2) row9643_eq_data branches
set_option maxRecDepth 4096 in
theorem row9649_eq_data : row9649 = Data.row9649 := rfl
def node9649 : Graph := .node 9649 row9649 node9643 Graph.empty
set_option maxRecDepth 4096 in
theorem node9649_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9649 :=
  Budget.node_covers _ _ 9649 row9649 node9643 Graph.empty
    node9643_covers empty_covers (by decide) (by decide) row9649_covers
theorem node9649_eq_data : node9649 = Data.node9649 := by
  have branches : (node9643, Graph.empty) = (Data.node9643, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9643_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9649 row children.1 children.2) row9649_eq_data branches
set_option maxRecDepth 4096 in
theorem row9677_eq_data : row9677 = Data.row9677 := rfl
def node9677 : Graph := .node 9677 row9677 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9677_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9677 :=
  Budget.node_covers _ _ 9677 row9677 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9677_covers
theorem node9677_eq_data : node9677 = Data.node9677 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9677 row children.1 children.2) row9677_eq_data branches
set_option maxRecDepth 4096 in
theorem row9679_eq_data : row9679 = Data.row9679 := rfl
def node9679 : Graph := .node 9679 row9679 node9677 Graph.empty
set_option maxRecDepth 4096 in
theorem node9679_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9679 :=
  Budget.node_covers _ _ 9679 row9679 node9677 Graph.empty
    node9677_covers empty_covers (by decide) (by decide) row9679_covers
theorem node9679_eq_data : node9679 = Data.node9679 := by
  have branches : (node9677, Graph.empty) = (Data.node9677, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9677_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9679 row children.1 children.2) row9679_eq_data branches
set_option maxRecDepth 4096 in
theorem row9661_eq_data : row9661 = Data.row9661 := rfl
def node9661 : Graph := .node 9661 row9661 node9649 node9679
set_option maxRecDepth 4096 in
theorem node9661_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9661 :=
  Budget.node_covers _ _ 9661 row9661 node9649 node9679
    node9649_covers node9679_covers (by decide) (by decide) row9661_covers
theorem node9661_eq_data : node9661 = Data.node9661 := by
  have branches : (node9649, node9679) = (Data.node9649, Data.node9679) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9649_eq_data node9679_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9661 row children.1 children.2) row9661_eq_data branches
set_option maxRecDepth 4096 in
theorem row9697_eq_data : row9697 = Data.row9697 := rfl
def node9697 : Graph := .node 9697 row9697 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9697_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9697 :=
  Budget.node_covers _ _ 9697 row9697 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9697_covers
theorem node9697_eq_data : node9697 = Data.node9697 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9697 row children.1 children.2) row9697_eq_data branches
set_option maxRecDepth 4096 in
theorem row9719_eq_data : row9719 = Data.row9719 := rfl
def node9719 : Graph := .node 9719 row9719 node9697 Graph.empty
set_option maxRecDepth 4096 in
theorem node9719_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9719 :=
  Budget.node_covers _ _ 9719 row9719 node9697 Graph.empty
    node9697_covers empty_covers (by decide) (by decide) row9719_covers
theorem node9719_eq_data : node9719 = Data.node9719 := by
  have branches : (node9697, Graph.empty) = (Data.node9697, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9697_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9719 row children.1 children.2) row9719_eq_data branches
set_option maxRecDepth 4096 in
theorem row9733_eq_data : row9733 = Data.row9733 := rfl
def node9733 : Graph := .node 9733 row9733 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9733_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9733 :=
  Budget.node_covers _ _ 9733 row9733 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9733_covers
theorem node9733_eq_data : node9733 = Data.node9733 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9733 row children.1 children.2) row9733_eq_data branches
set_option maxRecDepth 4096 in
theorem row9739_eq_data : row9739 = Data.row9739 := rfl
def node9739 : Graph := .node 9739 row9739 node9733 Graph.empty
set_option maxRecDepth 4096 in
theorem node9739_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9739 :=
  Budget.node_covers _ _ 9739 row9739 node9733 Graph.empty
    node9733_covers empty_covers (by decide) (by decide) row9739_covers
theorem node9739_eq_data : node9739 = Data.node9739 := by
  have branches : (node9733, Graph.empty) = (Data.node9733, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9733_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9739 row children.1 children.2) row9739_eq_data branches
set_option maxRecDepth 4096 in
theorem row9721_eq_data : row9721 = Data.row9721 := rfl
def node9721 : Graph := .node 9721 row9721 node9719 node9739
set_option maxRecDepth 4096 in
theorem node9721_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9721 :=
  Budget.node_covers _ _ 9721 row9721 node9719 node9739
    node9719_covers node9739_covers (by decide) (by decide) row9721_covers
theorem node9721_eq_data : node9721 = Data.node9721 := by
  have branches : (node9719, node9739) = (Data.node9719, Data.node9739) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9719_eq_data node9739_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9721 row children.1 children.2) row9721_eq_data branches
set_option maxRecDepth 4096 in
theorem row9689_eq_data : row9689 = Data.row9689 := rfl
def node9689 : Graph := .node 9689 row9689 node9661 node9721
set_option maxRecDepth 4096 in
theorem node9689_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9689 :=
  Budget.node_covers _ _ 9689 row9689 node9661 node9721
    node9661_covers node9721_covers (by decide) (by decide) row9689_covers
theorem node9689_eq_data : node9689 = Data.node9689 := by
  have branches : (node9661, node9721) = (Data.node9661, Data.node9721) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9661_eq_data node9721_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9689 row children.1 children.2) row9689_eq_data branches
set_option maxRecDepth 4096 in
theorem row9749_eq_data : row9749 = Data.row9749 := rfl
def node9749 : Graph := .node 9749 row9749 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9749_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9749 :=
  Budget.node_covers _ _ 9749 row9749 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9749_covers
theorem node9749_eq_data : node9749 = Data.node9749 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9749 row children.1 children.2) row9749_eq_data branches
set_option maxRecDepth 4096 in
theorem row9767_eq_data : row9767 = Data.row9767 := rfl
def node9767 : Graph := .node 9767 row9767 node9749 Graph.empty
set_option maxRecDepth 4096 in
theorem node9767_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9767 :=
  Budget.node_covers _ _ 9767 row9767 node9749 Graph.empty
    node9749_covers empty_covers (by decide) (by decide) row9767_covers
theorem node9767_eq_data : node9767 = Data.node9767 := by
  have branches : (node9749, Graph.empty) = (Data.node9749, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9749_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9767 row children.1 children.2) row9767_eq_data branches
set_option maxRecDepth 4096 in
theorem row9781_eq_data : row9781 = Data.row9781 := rfl
def node9781 : Graph := .node 9781 row9781 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9781_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9781 :=
  Budget.node_covers _ _ 9781 row9781 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9781_covers
theorem node9781_eq_data : node9781 = Data.node9781 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9781 row children.1 children.2) row9781_eq_data branches
set_option maxRecDepth 4096 in
theorem row9787_eq_data : row9787 = Data.row9787 := rfl
def node9787 : Graph := .node 9787 row9787 node9781 Graph.empty
set_option maxRecDepth 4096 in
theorem node9787_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9787 :=
  Budget.node_covers _ _ 9787 row9787 node9781 Graph.empty
    node9781_covers empty_covers (by decide) (by decide) row9787_covers
theorem node9787_eq_data : node9787 = Data.node9787 := by
  have branches : (node9781, Graph.empty) = (Data.node9781, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9781_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9787 row children.1 children.2) row9787_eq_data branches
set_option maxRecDepth 4096 in
theorem row9769_eq_data : row9769 = Data.row9769 := rfl
def node9769 : Graph := .node 9769 row9769 node9767 node9787
set_option maxRecDepth 4096 in
theorem node9769_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9769 :=
  Budget.node_covers _ _ 9769 row9769 node9767 node9787
    node9767_covers node9787_covers (by decide) (by decide) row9769_covers
theorem node9769_eq_data : node9769 = Data.node9769 := by
  have branches : (node9767, node9787) = (Data.node9767, Data.node9787) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9767_eq_data node9787_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9769 row children.1 children.2) row9769_eq_data branches
set_option maxRecDepth 4096 in
theorem row9803_eq_data : row9803 = Data.row9803 := rfl
def node9803 : Graph := .node 9803 row9803 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9803_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9803 :=
  Budget.node_covers _ _ 9803 row9803 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9803_covers
theorem node9803_eq_data : node9803 = Data.node9803 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9803 row children.1 children.2) row9803_eq_data branches
set_option maxRecDepth 4096 in
theorem row9811_eq_data : row9811 = Data.row9811 := rfl
def node9811 : Graph := .node 9811 row9811 node9803 Graph.empty
set_option maxRecDepth 4096 in
theorem node9811_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9811 :=
  Budget.node_covers _ _ 9811 row9811 node9803 Graph.empty
    node9803_covers empty_covers (by decide) (by decide) row9811_covers
theorem node9811_eq_data : node9811 = Data.node9811 := by
  have branches : (node9803, Graph.empty) = (Data.node9803, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9803_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9811 row children.1 children.2) row9811_eq_data branches
set_option maxRecDepth 4096 in
theorem row9829_eq_data : row9829 = Data.row9829 := rfl
def node9829 : Graph := .node 9829 row9829 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9829_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9829 :=
  Budget.node_covers _ _ 9829 row9829 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9829_covers
theorem node9829_eq_data : node9829 = Data.node9829 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9829 row children.1 children.2) row9829_eq_data branches
set_option maxRecDepth 4096 in
theorem row9833_eq_data : row9833 = Data.row9833 := rfl
def node9833 : Graph := .node 9833 row9833 node9829 Graph.empty
set_option maxRecDepth 4096 in
theorem node9833_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9833 :=
  Budget.node_covers _ _ 9833 row9833 node9829 Graph.empty
    node9829_covers empty_covers (by decide) (by decide) row9833_covers
theorem node9833_eq_data : node9833 = Data.node9833 := by
  have branches : (node9829, Graph.empty) = (Data.node9829, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9829_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9833 row children.1 children.2) row9833_eq_data branches
set_option maxRecDepth 4096 in
theorem row9817_eq_data : row9817 = Data.row9817 := rfl
def node9817 : Graph := .node 9817 row9817 node9811 node9833
set_option maxRecDepth 4096 in
theorem node9817_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9817 :=
  Budget.node_covers _ _ 9817 row9817 node9811 node9833
    node9811_covers node9833_covers (by decide) (by decide) row9817_covers
theorem node9817_eq_data : node9817 = Data.node9817 := by
  have branches : (node9811, node9833) = (Data.node9811, Data.node9833) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9811_eq_data node9833_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9817 row children.1 children.2) row9817_eq_data branches
set_option maxRecDepth 4096 in
theorem row9791_eq_data : row9791 = Data.row9791 := rfl
def node9791 : Graph := .node 9791 row9791 node9769 node9817
set_option maxRecDepth 4096 in
theorem node9791_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9791 :=
  Budget.node_covers _ _ 9791 row9791 node9769 node9817
    node9769_covers node9817_covers (by decide) (by decide) row9791_covers
theorem node9791_eq_data : node9791 = Data.node9791 := by
  have branches : (node9769, node9817) = (Data.node9769, Data.node9817) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9769_eq_data node9817_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9791 row children.1 children.2) row9791_eq_data branches
set_option maxRecDepth 4096 in
theorem row9743_eq_data : row9743 = Data.row9743 := rfl
def node9743 : Graph := .node 9743 row9743 node9689 node9791
set_option maxRecDepth 4096 in
theorem node9743_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9743 :=
  Budget.node_covers _ _ 9743 row9743 node9689 node9791
    node9689_covers node9791_covers (by decide) (by decide) row9743_covers
theorem node9743_eq_data : node9743 = Data.node9743 := by
  have branches : (node9689, node9791) = (Data.node9689, Data.node9791) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9689_eq_data node9791_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9743 row children.1 children.2) row9743_eq_data branches
set_option maxRecDepth 4096 in
theorem row9631_eq_data : row9631 = Data.row9631 := rfl
def node9631 : Graph := .node 9631 row9631 node9511 node9743
set_option maxRecDepth 4096 in
theorem node9631_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9631 :=
  Budget.node_covers _ _ 9631 row9631 node9511 node9743
    node9511_covers node9743_covers (by decide) (by decide) row9631_covers
theorem node9631_eq_data : node9631 = Data.node9631 := by
  have branches : (node9511, node9743) = (Data.node9511, Data.node9743) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9511_eq_data node9743_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9631 row children.1 children.2) row9631_eq_data branches
set_option maxRecDepth 4096 in
theorem row9851_eq_data : row9851 = Data.row9851 := rfl
def node9851 : Graph := .node 9851 row9851 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9851_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9851 :=
  Budget.node_covers _ _ 9851 row9851 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9851_covers
theorem node9851_eq_data : node9851 = Data.node9851 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9851 row children.1 children.2) row9851_eq_data branches
set_option maxRecDepth 4096 in
theorem row9857_eq_data : row9857 = Data.row9857 := rfl
def node9857 : Graph := .node 9857 row9857 node9851 Graph.empty
set_option maxRecDepth 4096 in
theorem node9857_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9857 :=
  Budget.node_covers _ _ 9857 row9857 node9851 Graph.empty
    node9851_covers empty_covers (by decide) (by decide) row9857_covers
theorem node9857_eq_data : node9857 = Data.node9857 := by
  have branches : (node9851, Graph.empty) = (Data.node9851, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9851_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9857 row children.1 children.2) row9857_eq_data branches
set_option maxRecDepth 4096 in
theorem row9871_eq_data : row9871 = Data.row9871 := rfl
def node9871 : Graph := .node 9871 row9871 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9871_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9871 :=
  Budget.node_covers _ _ 9871 row9871 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9871_covers
theorem node9871_eq_data : node9871 = Data.node9871 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9871 row children.1 children.2) row9871_eq_data branches
set_option maxRecDepth 4096 in
theorem row9883_eq_data : row9883 = Data.row9883 := rfl
def node9883 : Graph := .node 9883 row9883 node9871 Graph.empty
set_option maxRecDepth 4096 in
theorem node9883_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9883 :=
  Budget.node_covers _ _ 9883 row9883 node9871 Graph.empty
    node9871_covers empty_covers (by decide) (by decide) row9883_covers
theorem node9883_eq_data : node9883 = Data.node9883 := by
  have branches : (node9871, Graph.empty) = (Data.node9871, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9871_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9883 row children.1 children.2) row9883_eq_data branches
set_option maxRecDepth 4096 in
theorem row9859_eq_data : row9859 = Data.row9859 := rfl
def node9859 : Graph := .node 9859 row9859 node9857 node9883
set_option maxRecDepth 4096 in
theorem node9859_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9859 :=
  Budget.node_covers _ _ 9859 row9859 node9857 node9883
    node9857_covers node9883_covers (by decide) (by decide) row9859_covers
theorem node9859_eq_data : node9859 = Data.node9859 := by
  have branches : (node9857, node9883) = (Data.node9857, Data.node9883) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9857_eq_data node9883_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9859 row children.1 children.2) row9859_eq_data branches
set_option maxRecDepth 4096 in
theorem row9901_eq_data : row9901 = Data.row9901 := rfl
def node9901 : Graph := .node 9901 row9901 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node9901_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9901 :=
  Budget.node_covers _ _ 9901 row9901 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row9901_covers
theorem node9901_eq_data : node9901 = Data.node9901 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9901 row children.1 children.2) row9901_eq_data branches
set_option maxRecDepth 4096 in
theorem row9907_eq_data : row9907 = Data.row9907 := rfl
def node9907 : Graph := .node 9907 row9907 node9901 Graph.empty
set_option maxRecDepth 4096 in
theorem node9907_covers : Budget.Covers PrimePairVertices.largeTree 26034 node9907 :=
  Budget.node_covers _ _ 9907 row9907 node9901 Graph.empty
    node9901_covers empty_covers (by decide) (by decide) row9907_covers
theorem node9907_eq_data : node9907 = Data.node9907 := by
  have branches : (node9901, Graph.empty) = (Data.node9901, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node9901_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 9907 row children.1 children.2) row9907_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
