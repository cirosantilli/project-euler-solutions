import ProjectEulerSolutions.Lib.PrimePairLarge.Graph010
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row5399_eq_data : row5399 = Data.row5399 := rfl
def node5399 : Graph := .node 5399 row5399 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5399_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5399 :=
  Budget.node_covers _ _ 5399 row5399 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5399_covers
theorem node5399_eq_data : node5399 = Data.node5399 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5399 row children.1 children.2) row5399_eq_data branches
set_option maxRecDepth 4096 in
theorem row5407_eq_data : row5407 = Data.row5407 := rfl
def node5407 : Graph := .node 5407 row5407 node5399 Graph.empty
set_option maxRecDepth 4096 in
theorem node5407_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5407 :=
  Budget.node_covers _ _ 5407 row5407 node5399 Graph.empty
    node5399_covers empty_covers (by decide) (by decide) row5407_covers
theorem node5407_eq_data : node5407 = Data.node5407 := by
  have branches : (node5399, Graph.empty) = (Data.node5399, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5399_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5407 row children.1 children.2) row5407_eq_data branches
set_option maxRecDepth 4096 in
theorem row5417_eq_data : row5417 = Data.row5417 := rfl
def node5417 : Graph := .node 5417 row5417 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5417_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5417 :=
  Budget.node_covers _ _ 5417 row5417 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5417_covers
theorem node5417_eq_data : node5417 = Data.node5417 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5417 row children.1 children.2) row5417_eq_data branches
set_option maxRecDepth 4096 in
theorem row5419_eq_data : row5419 = Data.row5419 := rfl
def node5419 : Graph := .node 5419 row5419 node5417 Graph.empty
set_option maxRecDepth 4096 in
theorem node5419_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5419 :=
  Budget.node_covers _ _ 5419 row5419 node5417 Graph.empty
    node5417_covers empty_covers (by decide) (by decide) row5419_covers
theorem node5419_eq_data : node5419 = Data.node5419 := by
  have branches : (node5417, Graph.empty) = (Data.node5417, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5417_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5419 row children.1 children.2) row5419_eq_data branches
set_option maxRecDepth 4096 in
theorem row5413_eq_data : row5413 = Data.row5413 := rfl
def node5413 : Graph := .node 5413 row5413 node5407 node5419
set_option maxRecDepth 4096 in
theorem node5413_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5413 :=
  Budget.node_covers _ _ 5413 row5413 node5407 node5419
    node5407_covers node5419_covers (by decide) (by decide) row5413_covers
theorem node5413_eq_data : node5413 = Data.node5413 := by
  have branches : (node5407, node5419) = (Data.node5407, Data.node5419) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5407_eq_data node5419_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5413 row children.1 children.2) row5413_eq_data branches
set_option maxRecDepth 4096 in
theorem row5393_eq_data : row5393 = Data.row5393 := rfl
def node5393 : Graph := .node 5393 row5393 node5351 node5413
set_option maxRecDepth 4096 in
theorem node5393_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5393 :=
  Budget.node_covers _ _ 5393 row5393 node5351 node5413
    node5351_covers node5413_covers (by decide) (by decide) row5393_covers
theorem node5393_eq_data : node5393 = Data.node5393 := by
  have branches : (node5351, node5413) = (Data.node5351, Data.node5413) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5351_eq_data node5413_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5393 row children.1 children.2) row5393_eq_data branches
set_option maxRecDepth 4096 in
theorem row5437_eq_data : row5437 = Data.row5437 := rfl
def node5437 : Graph := .node 5437 row5437 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5437_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5437 :=
  Budget.node_covers _ _ 5437 row5437 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5437_covers
theorem node5437_eq_data : node5437 = Data.node5437 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5437 row children.1 children.2) row5437_eq_data branches
set_option maxRecDepth 4096 in
theorem row5441_eq_data : row5441 = Data.row5441 := rfl
def node5441 : Graph := .node 5441 row5441 node5437 Graph.empty
set_option maxRecDepth 4096 in
theorem node5441_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5441 :=
  Budget.node_covers _ _ 5441 row5441 node5437 Graph.empty
    node5437_covers empty_covers (by decide) (by decide) row5441_covers
theorem node5441_eq_data : node5441 = Data.node5441 := by
  have branches : (node5437, Graph.empty) = (Data.node5437, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5437_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5441 row children.1 children.2) row5441_eq_data branches
set_option maxRecDepth 4096 in
theorem row5449_eq_data : row5449 = Data.row5449 := rfl
def node5449 : Graph := .node 5449 row5449 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5449_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5449 :=
  Budget.node_covers _ _ 5449 row5449 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5449_covers
theorem node5449_eq_data : node5449 = Data.node5449 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5449 row children.1 children.2) row5449_eq_data branches
set_option maxRecDepth 4096 in
theorem row5471_eq_data : row5471 = Data.row5471 := rfl
def node5471 : Graph := .node 5471 row5471 node5449 Graph.empty
set_option maxRecDepth 4096 in
theorem node5471_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5471 :=
  Budget.node_covers _ _ 5471 row5471 node5449 Graph.empty
    node5449_covers empty_covers (by decide) (by decide) row5471_covers
theorem node5471_eq_data : node5471 = Data.node5471 := by
  have branches : (node5449, Graph.empty) = (Data.node5449, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5449_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5471 row children.1 children.2) row5471_eq_data branches
set_option maxRecDepth 4096 in
theorem row5443_eq_data : row5443 = Data.row5443 := rfl
def node5443 : Graph := .node 5443 row5443 node5441 node5471
set_option maxRecDepth 4096 in
theorem node5443_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5443 :=
  Budget.node_covers _ _ 5443 row5443 node5441 node5471
    node5441_covers node5471_covers (by decide) (by decide) row5443_covers
theorem node5443_eq_data : node5443 = Data.node5443 := by
  have branches : (node5441, node5471) = (Data.node5441, Data.node5471) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5441_eq_data node5471_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5443 row children.1 children.2) row5443_eq_data branches
set_option maxRecDepth 4096 in
theorem row5479_eq_data : row5479 = Data.row5479 := rfl
def node5479 : Graph := .node 5479 row5479 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5479_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5479 :=
  Budget.node_covers _ _ 5479 row5479 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5479_covers
theorem node5479_eq_data : node5479 = Data.node5479 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5479 row children.1 children.2) row5479_eq_data branches
set_option maxRecDepth 4096 in
theorem row5483_eq_data : row5483 = Data.row5483 := rfl
def node5483 : Graph := .node 5483 row5483 node5479 Graph.empty
set_option maxRecDepth 4096 in
theorem node5483_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5483 :=
  Budget.node_covers _ _ 5483 row5483 node5479 Graph.empty
    node5479_covers empty_covers (by decide) (by decide) row5483_covers
theorem node5483_eq_data : node5483 = Data.node5483 := by
  have branches : (node5479, Graph.empty) = (Data.node5479, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5479_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5483 row children.1 children.2) row5483_eq_data branches
set_option maxRecDepth 4096 in
theorem row5503_eq_data : row5503 = Data.row5503 := rfl
def node5503 : Graph := .node 5503 row5503 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5503_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5503 :=
  Budget.node_covers _ _ 5503 row5503 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5503_covers
theorem node5503_eq_data : node5503 = Data.node5503 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5503 row children.1 children.2) row5503_eq_data branches
set_option maxRecDepth 4096 in
theorem row5507_eq_data : row5507 = Data.row5507 := rfl
def node5507 : Graph := .node 5507 row5507 node5503 Graph.empty
set_option maxRecDepth 4096 in
theorem node5507_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5507 :=
  Budget.node_covers _ _ 5507 row5507 node5503 Graph.empty
    node5503_covers empty_covers (by decide) (by decide) row5507_covers
theorem node5507_eq_data : node5507 = Data.node5507 := by
  have branches : (node5503, Graph.empty) = (Data.node5503, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5503_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5507 row children.1 children.2) row5507_eq_data branches
set_option maxRecDepth 4096 in
theorem row5501_eq_data : row5501 = Data.row5501 := rfl
def node5501 : Graph := .node 5501 row5501 node5483 node5507
set_option maxRecDepth 4096 in
theorem node5501_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5501 :=
  Budget.node_covers _ _ 5501 row5501 node5483 node5507
    node5483_covers node5507_covers (by decide) (by decide) row5501_covers
theorem node5501_eq_data : node5501 = Data.node5501 := by
  have branches : (node5483, node5507) = (Data.node5483, Data.node5507) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5483_eq_data node5507_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5501 row children.1 children.2) row5501_eq_data branches
set_option maxRecDepth 4096 in
theorem row5477_eq_data : row5477 = Data.row5477 := rfl
def node5477 : Graph := .node 5477 row5477 node5443 node5501
set_option maxRecDepth 4096 in
theorem node5477_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5477 :=
  Budget.node_covers _ _ 5477 row5477 node5443 node5501
    node5443_covers node5501_covers (by decide) (by decide) row5477_covers
theorem node5477_eq_data : node5477 = Data.node5477 := by
  have branches : (node5443, node5501) = (Data.node5443, Data.node5501) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5443_eq_data node5501_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5477 row children.1 children.2) row5477_eq_data branches
set_option maxRecDepth 4096 in
theorem row5431_eq_data : row5431 = Data.row5431 := rfl
def node5431 : Graph := .node 5431 row5431 node5393 node5477
set_option maxRecDepth 4096 in
theorem node5431_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5431 :=
  Budget.node_covers _ _ 5431 row5431 node5393 node5477
    node5393_covers node5477_covers (by decide) (by decide) row5431_covers
theorem node5431_eq_data : node5431 = Data.node5431 := by
  have branches : (node5393, node5477) = (Data.node5393, Data.node5477) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5393_eq_data node5477_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5431 row children.1 children.2) row5431_eq_data branches
set_option maxRecDepth 4096 in
theorem row5323_eq_data : row5323 = Data.row5323 := rfl
def node5323 : Graph := .node 5323 row5323 node5209 node5431
set_option maxRecDepth 4096 in
theorem node5323_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5323 :=
  Budget.node_covers _ _ 5323 row5323 node5209 node5431
    node5209_covers node5431_covers (by decide) (by decide) row5323_covers
theorem node5323_eq_data : node5323 = Data.node5323 := by
  have branches : (node5209, node5431) = (Data.node5209, Data.node5431) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5209_eq_data node5431_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5323 row children.1 children.2) row5323_eq_data branches
set_option maxRecDepth 4096 in
theorem row5521_eq_data : row5521 = Data.row5521 := rfl
def node5521 : Graph := .node 5521 row5521 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5521_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5521 :=
  Budget.node_covers _ _ 5521 row5521 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5521_covers
theorem node5521_eq_data : node5521 = Data.node5521 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5521 row children.1 children.2) row5521_eq_data branches
set_option maxRecDepth 4096 in
theorem row5527_eq_data : row5527 = Data.row5527 := rfl
def node5527 : Graph := .node 5527 row5527 node5521 Graph.empty
set_option maxRecDepth 4096 in
theorem node5527_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5527 :=
  Budget.node_covers _ _ 5527 row5527 node5521 Graph.empty
    node5521_covers empty_covers (by decide) (by decide) row5527_covers
theorem node5527_eq_data : node5527 = Data.node5527 := by
  have branches : (node5521, Graph.empty) = (Data.node5521, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5521_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5527 row children.1 children.2) row5527_eq_data branches
set_option maxRecDepth 4096 in
theorem row5557_eq_data : row5557 = Data.row5557 := rfl
def node5557 : Graph := .node 5557 row5557 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5557_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5557 :=
  Budget.node_covers _ _ 5557 row5557 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5557_covers
theorem node5557_eq_data : node5557 = Data.node5557 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5557 row children.1 children.2) row5557_eq_data branches
set_option maxRecDepth 4096 in
theorem row5563_eq_data : row5563 = Data.row5563 := rfl
def node5563 : Graph := .node 5563 row5563 node5557 Graph.empty
set_option maxRecDepth 4096 in
theorem node5563_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5563 :=
  Budget.node_covers _ _ 5563 row5563 node5557 Graph.empty
    node5557_covers empty_covers (by decide) (by decide) row5563_covers
theorem node5563_eq_data : node5563 = Data.node5563 := by
  have branches : (node5557, Graph.empty) = (Data.node5557, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5557_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5563 row children.1 children.2) row5563_eq_data branches
set_option maxRecDepth 4096 in
theorem row5531_eq_data : row5531 = Data.row5531 := rfl
def node5531 : Graph := .node 5531 row5531 node5527 node5563
set_option maxRecDepth 4096 in
theorem node5531_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5531 :=
  Budget.node_covers _ _ 5531 row5531 node5527 node5563
    node5527_covers node5563_covers (by decide) (by decide) row5531_covers
theorem node5531_eq_data : node5531 = Data.node5531 := by
  have branches : (node5527, node5563) = (Data.node5527, Data.node5563) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5527_eq_data node5563_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5531 row children.1 children.2) row5531_eq_data branches
set_option maxRecDepth 4096 in
theorem row5573_eq_data : row5573 = Data.row5573 := rfl
def node5573 : Graph := .node 5573 row5573 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5573_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5573 :=
  Budget.node_covers _ _ 5573 row5573 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5573_covers
theorem node5573_eq_data : node5573 = Data.node5573 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5573 row children.1 children.2) row5573_eq_data branches
set_option maxRecDepth 4096 in
theorem row5581_eq_data : row5581 = Data.row5581 := rfl
def node5581 : Graph := .node 5581 row5581 node5573 Graph.empty
set_option maxRecDepth 4096 in
theorem node5581_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5581 :=
  Budget.node_covers _ _ 5581 row5581 node5573 Graph.empty
    node5573_covers empty_covers (by decide) (by decide) row5581_covers
theorem node5581_eq_data : node5581 = Data.node5581 := by
  have branches : (node5573, Graph.empty) = (Data.node5573, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5573_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5581 row children.1 children.2) row5581_eq_data branches
set_option maxRecDepth 4096 in
theorem row5623_eq_data : row5623 = Data.row5623 := rfl
def node5623 : Graph := .node 5623 row5623 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5623_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5623 :=
  Budget.node_covers _ _ 5623 row5623 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5623_covers
theorem node5623_eq_data : node5623 = Data.node5623 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5623 row children.1 children.2) row5623_eq_data branches
set_option maxRecDepth 4096 in
theorem row5639_eq_data : row5639 = Data.row5639 := rfl
def node5639 : Graph := .node 5639 row5639 node5623 Graph.empty
set_option maxRecDepth 4096 in
theorem node5639_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5639 :=
  Budget.node_covers _ _ 5639 row5639 node5623 Graph.empty
    node5623_covers empty_covers (by decide) (by decide) row5639_covers
theorem node5639_eq_data : node5639 = Data.node5639 := by
  have branches : (node5623, Graph.empty) = (Data.node5623, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5623_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5639 row children.1 children.2) row5639_eq_data branches
set_option maxRecDepth 4096 in
theorem row5591_eq_data : row5591 = Data.row5591 := rfl
def node5591 : Graph := .node 5591 row5591 node5581 node5639
set_option maxRecDepth 4096 in
theorem node5591_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5591 :=
  Budget.node_covers _ _ 5591 row5591 node5581 node5639
    node5581_covers node5639_covers (by decide) (by decide) row5591_covers
theorem node5591_eq_data : node5591 = Data.node5591 := by
  have branches : (node5581, node5639) = (Data.node5581, Data.node5639) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5581_eq_data node5639_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5591 row children.1 children.2) row5591_eq_data branches
set_option maxRecDepth 4096 in
theorem row5569_eq_data : row5569 = Data.row5569 := rfl
def node5569 : Graph := .node 5569 row5569 node5531 node5591
set_option maxRecDepth 4096 in
theorem node5569_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5569 :=
  Budget.node_covers _ _ 5569 row5569 node5531 node5591
    node5531_covers node5591_covers (by decide) (by decide) row5569_covers
theorem node5569_eq_data : node5569 = Data.node5569 := by
  have branches : (node5531, node5591) = (Data.node5531, Data.node5591) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5531_eq_data node5591_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5569 row children.1 children.2) row5569_eq_data branches
set_option maxRecDepth 4096 in
theorem row5647_eq_data : row5647 = Data.row5647 := rfl
def node5647 : Graph := .node 5647 row5647 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5647_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5647 :=
  Budget.node_covers _ _ 5647 row5647 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5647_covers
theorem node5647_eq_data : node5647 = Data.node5647 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5647 row children.1 children.2) row5647_eq_data branches
set_option maxRecDepth 4096 in
theorem row5651_eq_data : row5651 = Data.row5651 := rfl
def node5651 : Graph := .node 5651 row5651 node5647 Graph.empty
set_option maxRecDepth 4096 in
theorem node5651_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5651 :=
  Budget.node_covers _ _ 5651 row5651 node5647 Graph.empty
    node5647_covers empty_covers (by decide) (by decide) row5651_covers
theorem node5651_eq_data : node5651 = Data.node5651 := by
  have branches : (node5647, Graph.empty) = (Data.node5647, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5647_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5651 row children.1 children.2) row5651_eq_data branches
set_option maxRecDepth 4096 in
theorem row5657_eq_data : row5657 = Data.row5657 := rfl
def node5657 : Graph := .node 5657 row5657 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5657_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5657 :=
  Budget.node_covers _ _ 5657 row5657 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5657_covers
theorem node5657_eq_data : node5657 = Data.node5657 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5657 row children.1 children.2) row5657_eq_data branches
set_option maxRecDepth 4096 in
theorem row5659_eq_data : row5659 = Data.row5659 := rfl
def node5659 : Graph := .node 5659 row5659 node5657 Graph.empty
set_option maxRecDepth 4096 in
theorem node5659_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5659 :=
  Budget.node_covers _ _ 5659 row5659 node5657 Graph.empty
    node5657_covers empty_covers (by decide) (by decide) row5659_covers
theorem node5659_eq_data : node5659 = Data.node5659 := by
  have branches : (node5657, Graph.empty) = (Data.node5657, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5657_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5659 row children.1 children.2) row5659_eq_data branches
set_option maxRecDepth 4096 in
theorem row5653_eq_data : row5653 = Data.row5653 := rfl
def node5653 : Graph := .node 5653 row5653 node5651 node5659
set_option maxRecDepth 4096 in
theorem node5653_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5653 :=
  Budget.node_covers _ _ 5653 row5653 node5651 node5659
    node5651_covers node5659_covers (by decide) (by decide) row5653_covers
theorem node5653_eq_data : node5653 = Data.node5653 := by
  have branches : (node5651, node5659) = (Data.node5651, Data.node5659) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5651_eq_data node5659_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5653 row children.1 children.2) row5653_eq_data branches
set_option maxRecDepth 4096 in
theorem row5683_eq_data : row5683 = Data.row5683 := rfl
def node5683 : Graph := .node 5683 row5683 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5683_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5683 :=
  Budget.node_covers _ _ 5683 row5683 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5683_covers
theorem node5683_eq_data : node5683 = Data.node5683 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5683 row children.1 children.2) row5683_eq_data branches
set_option maxRecDepth 4096 in
theorem row5689_eq_data : row5689 = Data.row5689 := rfl
def node5689 : Graph := .node 5689 row5689 node5683 Graph.empty
set_option maxRecDepth 4096 in
theorem node5689_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5689 :=
  Budget.node_covers _ _ 5689 row5689 node5683 Graph.empty
    node5683_covers empty_covers (by decide) (by decide) row5689_covers
theorem node5689_eq_data : node5689 = Data.node5689 := by
  have branches : (node5683, Graph.empty) = (Data.node5683, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5683_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5689 row children.1 children.2) row5689_eq_data branches
set_option maxRecDepth 4096 in
theorem row5701_eq_data : row5701 = Data.row5701 := rfl
def node5701 : Graph := .node 5701 row5701 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5701_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5701 :=
  Budget.node_covers _ _ 5701 row5701 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5701_covers
theorem node5701_eq_data : node5701 = Data.node5701 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5701 row children.1 children.2) row5701_eq_data branches
set_option maxRecDepth 4096 in
theorem row5711_eq_data : row5711 = Data.row5711 := rfl
def node5711 : Graph := .node 5711 row5711 node5701 Graph.empty
set_option maxRecDepth 4096 in
theorem node5711_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5711 :=
  Budget.node_covers _ _ 5711 row5711 node5701 Graph.empty
    node5701_covers empty_covers (by decide) (by decide) row5711_covers
theorem node5711_eq_data : node5711 = Data.node5711 := by
  have branches : (node5701, Graph.empty) = (Data.node5701, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5701_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5711 row children.1 children.2) row5711_eq_data branches
set_option maxRecDepth 4096 in
theorem row5693_eq_data : row5693 = Data.row5693 := rfl
def node5693 : Graph := .node 5693 row5693 node5689 node5711
set_option maxRecDepth 4096 in
theorem node5693_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5693 :=
  Budget.node_covers _ _ 5693 row5693 node5689 node5711
    node5689_covers node5711_covers (by decide) (by decide) row5693_covers
theorem node5693_eq_data : node5693 = Data.node5693 := by
  have branches : (node5689, node5711) = (Data.node5689, Data.node5711) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5689_eq_data node5711_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5693 row children.1 children.2) row5693_eq_data branches
set_option maxRecDepth 4096 in
theorem row5669_eq_data : row5669 = Data.row5669 := rfl
def node5669 : Graph := .node 5669 row5669 node5653 node5693
set_option maxRecDepth 4096 in
theorem node5669_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5669 :=
  Budget.node_covers _ _ 5669 row5669 node5653 node5693
    node5653_covers node5693_covers (by decide) (by decide) row5669_covers
theorem node5669_eq_data : node5669 = Data.node5669 := by
  have branches : (node5653, node5693) = (Data.node5653, Data.node5693) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5653_eq_data node5693_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5669 row children.1 children.2) row5669_eq_data branches
set_option maxRecDepth 4096 in
theorem row5641_eq_data : row5641 = Data.row5641 := rfl
def node5641 : Graph := .node 5641 row5641 node5569 node5669
set_option maxRecDepth 4096 in
theorem node5641_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5641 :=
  Budget.node_covers _ _ 5641 row5641 node5569 node5669
    node5569_covers node5669_covers (by decide) (by decide) row5641_covers
theorem node5641_eq_data : node5641 = Data.node5641 := by
  have branches : (node5569, node5669) = (Data.node5569, Data.node5669) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5569_eq_data node5669_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5641 row children.1 children.2) row5641_eq_data branches
set_option maxRecDepth 4096 in
theorem row5737_eq_data : row5737 = Data.row5737 := rfl
def node5737 : Graph := .node 5737 row5737 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5737_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5737 :=
  Budget.node_covers _ _ 5737 row5737 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5737_covers
theorem node5737_eq_data : node5737 = Data.node5737 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5737 row children.1 children.2) row5737_eq_data branches
set_option maxRecDepth 4096 in
theorem row5741_eq_data : row5741 = Data.row5741 := rfl
def node5741 : Graph := .node 5741 row5741 node5737 Graph.empty
set_option maxRecDepth 4096 in
theorem node5741_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5741 :=
  Budget.node_covers _ _ 5741 row5741 node5737 Graph.empty
    node5737_covers empty_covers (by decide) (by decide) row5741_covers
theorem node5741_eq_data : node5741 = Data.node5741 := by
  have branches : (node5737, Graph.empty) = (Data.node5737, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5737_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5741 row children.1 children.2) row5741_eq_data branches
set_option maxRecDepth 4096 in
theorem row5749_eq_data : row5749 = Data.row5749 := rfl
def node5749 : Graph := .node 5749 row5749 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5749_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5749 :=
  Budget.node_covers _ _ 5749 row5749 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5749_covers
theorem node5749_eq_data : node5749 = Data.node5749 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5749 row children.1 children.2) row5749_eq_data branches
set_option maxRecDepth 4096 in
theorem row5779_eq_data : row5779 = Data.row5779 := rfl
def node5779 : Graph := .node 5779 row5779 node5749 Graph.empty
set_option maxRecDepth 4096 in
theorem node5779_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5779 :=
  Budget.node_covers _ _ 5779 row5779 node5749 Graph.empty
    node5749_covers empty_covers (by decide) (by decide) row5779_covers
theorem node5779_eq_data : node5779 = Data.node5779 := by
  have branches : (node5749, Graph.empty) = (Data.node5749, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5749_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5779 row children.1 children.2) row5779_eq_data branches
set_option maxRecDepth 4096 in
theorem row5743_eq_data : row5743 = Data.row5743 := rfl
def node5743 : Graph := .node 5743 row5743 node5741 node5779
set_option maxRecDepth 4096 in
theorem node5743_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5743 :=
  Budget.node_covers _ _ 5743 row5743 node5741 node5779
    node5741_covers node5779_covers (by decide) (by decide) row5743_covers
theorem node5743_eq_data : node5743 = Data.node5743 := by
  have branches : (node5741, node5779) = (Data.node5741, Data.node5779) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5741_eq_data node5779_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5743 row children.1 children.2) row5743_eq_data branches
set_option maxRecDepth 4096 in
theorem row5791_eq_data : row5791 = Data.row5791 := rfl
def node5791 : Graph := .node 5791 row5791 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5791_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5791 :=
  Budget.node_covers _ _ 5791 row5791 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5791_covers
theorem node5791_eq_data : node5791 = Data.node5791 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5791 row children.1 children.2) row5791_eq_data branches
set_option maxRecDepth 4096 in
theorem row5801_eq_data : row5801 = Data.row5801 := rfl
def node5801 : Graph := .node 5801 row5801 node5791 Graph.empty
set_option maxRecDepth 4096 in
theorem node5801_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5801 :=
  Budget.node_covers _ _ 5801 row5801 node5791 Graph.empty
    node5791_covers empty_covers (by decide) (by decide) row5801_covers
theorem node5801_eq_data : node5801 = Data.node5801 := by
  have branches : (node5791, Graph.empty) = (Data.node5791, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5791_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5801 row children.1 children.2) row5801_eq_data branches
set_option maxRecDepth 4096 in
theorem row5813_eq_data : row5813 = Data.row5813 := rfl
def node5813 : Graph := .node 5813 row5813 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5813_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5813 :=
  Budget.node_covers _ _ 5813 row5813 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5813_covers
theorem node5813_eq_data : node5813 = Data.node5813 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5813 row children.1 children.2) row5813_eq_data branches
set_option maxRecDepth 4096 in
theorem row5821_eq_data : row5821 = Data.row5821 := rfl
def node5821 : Graph := .node 5821 row5821 node5813 Graph.empty
set_option maxRecDepth 4096 in
theorem node5821_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5821 :=
  Budget.node_covers _ _ 5821 row5821 node5813 Graph.empty
    node5813_covers empty_covers (by decide) (by decide) row5821_covers
theorem node5821_eq_data : node5821 = Data.node5821 := by
  have branches : (node5813, Graph.empty) = (Data.node5813, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5813_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5821 row children.1 children.2) row5821_eq_data branches
set_option maxRecDepth 4096 in
theorem row5807_eq_data : row5807 = Data.row5807 := rfl
def node5807 : Graph := .node 5807 row5807 node5801 node5821
set_option maxRecDepth 4096 in
theorem node5807_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5807 :=
  Budget.node_covers _ _ 5807 row5807 node5801 node5821
    node5801_covers node5821_covers (by decide) (by decide) row5807_covers
theorem node5807_eq_data : node5807 = Data.node5807 := by
  have branches : (node5801, node5821) = (Data.node5801, Data.node5821) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5801_eq_data node5821_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5807 row children.1 children.2) row5807_eq_data branches
set_option maxRecDepth 4096 in
theorem row5783_eq_data : row5783 = Data.row5783 := rfl
def node5783 : Graph := .node 5783 row5783 node5743 node5807
set_option maxRecDepth 4096 in
theorem node5783_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5783 :=
  Budget.node_covers _ _ 5783 row5783 node5743 node5807
    node5743_covers node5807_covers (by decide) (by decide) row5783_covers
theorem node5783_eq_data : node5783 = Data.node5783 := by
  have branches : (node5743, node5807) = (Data.node5743, Data.node5807) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5743_eq_data node5807_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5783 row children.1 children.2) row5783_eq_data branches
set_option maxRecDepth 4096 in
theorem row5839_eq_data : row5839 = Data.row5839 := rfl
def node5839 : Graph := .node 5839 row5839 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5839_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5839 :=
  Budget.node_covers _ _ 5839 row5839 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5839_covers
theorem node5839_eq_data : node5839 = Data.node5839 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5839 row children.1 children.2) row5839_eq_data branches
set_option maxRecDepth 4096 in
theorem row5843_eq_data : row5843 = Data.row5843 := rfl
def node5843 : Graph := .node 5843 row5843 node5839 Graph.empty
set_option maxRecDepth 4096 in
theorem node5843_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5843 :=
  Budget.node_covers _ _ 5843 row5843 node5839 Graph.empty
    node5839_covers empty_covers (by decide) (by decide) row5843_covers
theorem node5843_eq_data : node5843 = Data.node5843 := by
  have branches : (node5839, Graph.empty) = (Data.node5839, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5839_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5843 row children.1 children.2) row5843_eq_data branches
set_option maxRecDepth 4096 in
theorem row5851_eq_data : row5851 = Data.row5851 := rfl
def node5851 : Graph := .node 5851 row5851 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5851_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5851 :=
  Budget.node_covers _ _ 5851 row5851 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5851_covers
theorem node5851_eq_data : node5851 = Data.node5851 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5851 row children.1 children.2) row5851_eq_data branches
set_option maxRecDepth 4096 in
theorem row5857_eq_data : row5857 = Data.row5857 := rfl
def node5857 : Graph := .node 5857 row5857 node5851 Graph.empty
set_option maxRecDepth 4096 in
theorem node5857_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5857 :=
  Budget.node_covers _ _ 5857 row5857 node5851 Graph.empty
    node5851_covers empty_covers (by decide) (by decide) row5857_covers
theorem node5857_eq_data : node5857 = Data.node5857 := by
  have branches : (node5851, Graph.empty) = (Data.node5851, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5851_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5857 row children.1 children.2) row5857_eq_data branches
set_option maxRecDepth 4096 in
theorem row5849_eq_data : row5849 = Data.row5849 := rfl
def node5849 : Graph := .node 5849 row5849 node5843 node5857
set_option maxRecDepth 4096 in
theorem node5849_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5849 :=
  Budget.node_covers _ _ 5849 row5849 node5843 node5857
    node5843_covers node5857_covers (by decide) (by decide) row5849_covers
theorem node5849_eq_data : node5849 = Data.node5849 := by
  have branches : (node5843, node5857) = (Data.node5843, Data.node5857) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5843_eq_data node5857_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5849 row children.1 children.2) row5849_eq_data branches
set_option maxRecDepth 4096 in
theorem row5867_eq_data : row5867 = Data.row5867 := rfl
def node5867 : Graph := .node 5867 row5867 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5867_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5867 :=
  Budget.node_covers _ _ 5867 row5867 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5867_covers
theorem node5867_eq_data : node5867 = Data.node5867 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5867 row children.1 children.2) row5867_eq_data branches
set_option maxRecDepth 4096 in
theorem row5869_eq_data : row5869 = Data.row5869 := rfl
def node5869 : Graph := .node 5869 row5869 node5867 Graph.empty
set_option maxRecDepth 4096 in
theorem node5869_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5869 :=
  Budget.node_covers _ _ 5869 row5869 node5867 Graph.empty
    node5867_covers empty_covers (by decide) (by decide) row5869_covers
theorem node5869_eq_data : node5869 = Data.node5869 := by
  have branches : (node5867, Graph.empty) = (Data.node5867, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5867_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5869 row children.1 children.2) row5869_eq_data branches
set_option maxRecDepth 4096 in
theorem row5881_eq_data : row5881 = Data.row5881 := rfl
def node5881 : Graph := .node 5881 row5881 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node5881_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5881 :=
  Budget.node_covers _ _ 5881 row5881 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row5881_covers
theorem node5881_eq_data : node5881 = Data.node5881 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5881 row children.1 children.2) row5881_eq_data branches
set_option maxRecDepth 4096 in
theorem row5897_eq_data : row5897 = Data.row5897 := rfl
def node5897 : Graph := .node 5897 row5897 node5881 Graph.empty
set_option maxRecDepth 4096 in
theorem node5897_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5897 :=
  Budget.node_covers _ _ 5897 row5897 node5881 Graph.empty
    node5881_covers empty_covers (by decide) (by decide) row5897_covers
theorem node5897_eq_data : node5897 = Data.node5897 := by
  have branches : (node5881, Graph.empty) = (Data.node5881, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5881_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5897 row children.1 children.2) row5897_eq_data branches
set_option maxRecDepth 4096 in
theorem row5879_eq_data : row5879 = Data.row5879 := rfl
def node5879 : Graph := .node 5879 row5879 node5869 node5897
set_option maxRecDepth 4096 in
theorem node5879_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5879 :=
  Budget.node_covers _ _ 5879 row5879 node5869 node5897
    node5869_covers node5897_covers (by decide) (by decide) row5879_covers
theorem node5879_eq_data : node5879 = Data.node5879 := by
  have branches : (node5869, node5897) = (Data.node5869, Data.node5897) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5869_eq_data node5897_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5879 row children.1 children.2) row5879_eq_data branches
set_option maxRecDepth 4096 in
theorem row5861_eq_data : row5861 = Data.row5861 := rfl
def node5861 : Graph := .node 5861 row5861 node5849 node5879
set_option maxRecDepth 4096 in
theorem node5861_covers : Budget.Covers PrimePairVertices.largeTree 26034 node5861 :=
  Budget.node_covers _ _ 5861 row5861 node5849 node5879
    node5849_covers node5879_covers (by decide) (by decide) row5861_covers
theorem node5861_eq_data : node5861 = Data.node5861 := by
  have branches : (node5849, node5879) = (Data.node5849, Data.node5879) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node5849_eq_data node5879_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 5861 row children.1 children.2) row5861_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
