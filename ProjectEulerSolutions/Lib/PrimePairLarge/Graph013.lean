import ProjectEulerSolutions.Lib.PrimePairLarge.Graph012
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row6379_eq_data : row6379 = Data.row6379 := rfl
def node6379 : Graph := .node 6379 row6379 node6361 node6421
set_option maxRecDepth 4096 in
theorem node6379_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6379 :=
  Budget.node_covers _ _ 6379 row6379 node6361 node6421
    node6361_covers node6421_covers (by decide) (by decide) row6379_covers
theorem node6379_eq_data : node6379 = Data.node6379 := by
  have branches : (node6361, node6421) = (Data.node6361, Data.node6421) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6361_eq_data node6421_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6379 row children.1 children.2) row6379_eq_data branches
set_option maxRecDepth 4096 in
theorem row6469_eq_data : row6469 = Data.row6469 := rfl
def node6469 : Graph := .node 6469 row6469 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6469_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6469 :=
  Budget.node_covers _ _ 6469 row6469 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6469_covers
theorem node6469_eq_data : node6469 = Data.node6469 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6469 row children.1 children.2) row6469_eq_data branches
set_option maxRecDepth 4096 in
theorem row6473_eq_data : row6473 = Data.row6473 := rfl
def node6473 : Graph := .node 6473 row6473 node6469 Graph.empty
set_option maxRecDepth 4096 in
theorem node6473_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6473 :=
  Budget.node_covers _ _ 6473 row6473 node6469 Graph.empty
    node6469_covers empty_covers (by decide) (by decide) row6473_covers
theorem node6473_eq_data : node6473 = Data.node6473 := by
  have branches : (node6469, Graph.empty) = (Data.node6469, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6469_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6473 row children.1 children.2) row6473_eq_data branches
set_option maxRecDepth 4096 in
theorem row6491_eq_data : row6491 = Data.row6491 := rfl
def node6491 : Graph := .node 6491 row6491 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6491_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6491 :=
  Budget.node_covers _ _ 6491 row6491 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6491_covers
theorem node6491_eq_data : node6491 = Data.node6491 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6491 row children.1 children.2) row6491_eq_data branches
set_option maxRecDepth 4096 in
theorem row6521_eq_data : row6521 = Data.row6521 := rfl
def node6521 : Graph := .node 6521 row6521 node6491 Graph.empty
set_option maxRecDepth 4096 in
theorem node6521_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6521 :=
  Budget.node_covers _ _ 6521 row6521 node6491 Graph.empty
    node6491_covers empty_covers (by decide) (by decide) row6521_covers
theorem node6521_eq_data : node6521 = Data.node6521 := by
  have branches : (node6491, Graph.empty) = (Data.node6491, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6491_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6521 row children.1 children.2) row6521_eq_data branches
set_option maxRecDepth 4096 in
theorem row6481_eq_data : row6481 = Data.row6481 := rfl
def node6481 : Graph := .node 6481 row6481 node6473 node6521
set_option maxRecDepth 4096 in
theorem node6481_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6481 :=
  Budget.node_covers _ _ 6481 row6481 node6473 node6521
    node6473_covers node6521_covers (by decide) (by decide) row6481_covers
theorem node6481_eq_data : node6481 = Data.node6481 := by
  have branches : (node6473, node6521) = (Data.node6473, Data.node6521) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6473_eq_data node6521_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6481 row children.1 children.2) row6481_eq_data branches
set_option maxRecDepth 4096 in
theorem row6547_eq_data : row6547 = Data.row6547 := rfl
def node6547 : Graph := .node 6547 row6547 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6547_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6547 :=
  Budget.node_covers _ _ 6547 row6547 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6547_covers
theorem node6547_eq_data : node6547 = Data.node6547 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6547 row children.1 children.2) row6547_eq_data branches
set_option maxRecDepth 4096 in
theorem row6551_eq_data : row6551 = Data.row6551 := rfl
def node6551 : Graph := .node 6551 row6551 node6547 Graph.empty
set_option maxRecDepth 4096 in
theorem node6551_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6551 :=
  Budget.node_covers _ _ 6551 row6551 node6547 Graph.empty
    node6547_covers empty_covers (by decide) (by decide) row6551_covers
theorem node6551_eq_data : node6551 = Data.node6551 := by
  have branches : (node6547, Graph.empty) = (Data.node6547, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6547_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6551 row children.1 children.2) row6551_eq_data branches
set_option maxRecDepth 4096 in
theorem row6563_eq_data : row6563 = Data.row6563 := rfl
def node6563 : Graph := .node 6563 row6563 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6563_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6563 :=
  Budget.node_covers _ _ 6563 row6563 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6563_covers
theorem node6563_eq_data : node6563 = Data.node6563 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6563 row children.1 children.2) row6563_eq_data branches
set_option maxRecDepth 4096 in
theorem row6569_eq_data : row6569 = Data.row6569 := rfl
def node6569 : Graph := .node 6569 row6569 node6563 Graph.empty
set_option maxRecDepth 4096 in
theorem node6569_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6569 :=
  Budget.node_covers _ _ 6569 row6569 node6563 Graph.empty
    node6563_covers empty_covers (by decide) (by decide) row6569_covers
theorem node6569_eq_data : node6569 = Data.node6569 := by
  have branches : (node6563, Graph.empty) = (Data.node6563, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6563_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6569 row children.1 children.2) row6569_eq_data branches
set_option maxRecDepth 4096 in
theorem row6553_eq_data : row6553 = Data.row6553 := rfl
def node6553 : Graph := .node 6553 row6553 node6551 node6569
set_option maxRecDepth 4096 in
theorem node6553_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6553 :=
  Budget.node_covers _ _ 6553 row6553 node6551 node6569
    node6551_covers node6569_covers (by decide) (by decide) row6553_covers
theorem node6553_eq_data : node6553 = Data.node6553 := by
  have branches : (node6551, node6569) = (Data.node6551, Data.node6569) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6551_eq_data node6569_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6553 row children.1 children.2) row6553_eq_data branches
set_option maxRecDepth 4096 in
theorem row6529_eq_data : row6529 = Data.row6529 := rfl
def node6529 : Graph := .node 6529 row6529 node6481 node6553
set_option maxRecDepth 4096 in
theorem node6529_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6529 :=
  Budget.node_covers _ _ 6529 row6529 node6481 node6553
    node6481_covers node6553_covers (by decide) (by decide) row6529_covers
theorem node6529_eq_data : node6529 = Data.node6529 := by
  have branches : (node6481, node6553) = (Data.node6481, Data.node6553) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6481_eq_data node6553_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6529 row children.1 children.2) row6529_eq_data branches
set_option maxRecDepth 4096 in
theorem row6451_eq_data : row6451 = Data.row6451 := rfl
def node6451 : Graph := .node 6451 row6451 node6379 node6529
set_option maxRecDepth 4096 in
theorem node6451_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6451 :=
  Budget.node_covers _ _ 6451 row6451 node6379 node6529
    node6379_covers node6529_covers (by decide) (by decide) row6451_covers
theorem node6451_eq_data : node6451 = Data.node6451 := by
  have branches : (node6379, node6529) = (Data.node6379, Data.node6529) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6379_eq_data node6529_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6451 row children.1 children.2) row6451_eq_data branches
set_option maxRecDepth 4096 in
theorem row6577_eq_data : row6577 = Data.row6577 := rfl
def node6577 : Graph := .node 6577 row6577 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6577_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6577 :=
  Budget.node_covers _ _ 6577 row6577 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6577_covers
theorem node6577_eq_data : node6577 = Data.node6577 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6577 row children.1 children.2) row6577_eq_data branches
set_option maxRecDepth 4096 in
theorem row6581_eq_data : row6581 = Data.row6581 := rfl
def node6581 : Graph := .node 6581 row6581 node6577 Graph.empty
set_option maxRecDepth 4096 in
theorem node6581_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6581 :=
  Budget.node_covers _ _ 6581 row6581 node6577 Graph.empty
    node6577_covers empty_covers (by decide) (by decide) row6581_covers
theorem node6581_eq_data : node6581 = Data.node6581 := by
  have branches : (node6577, Graph.empty) = (Data.node6577, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6577_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6581 row children.1 children.2) row6581_eq_data branches
set_option maxRecDepth 4096 in
theorem row6607_eq_data : row6607 = Data.row6607 := rfl
def node6607 : Graph := .node 6607 row6607 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6607_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6607 :=
  Budget.node_covers _ _ 6607 row6607 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6607_covers
theorem node6607_eq_data : node6607 = Data.node6607 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6607 row children.1 children.2) row6607_eq_data branches
set_option maxRecDepth 4096 in
theorem row6619_eq_data : row6619 = Data.row6619 := rfl
def node6619 : Graph := .node 6619 row6619 node6607 Graph.empty
set_option maxRecDepth 4096 in
theorem node6619_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6619 :=
  Budget.node_covers _ _ 6619 row6619 node6607 Graph.empty
    node6607_covers empty_covers (by decide) (by decide) row6619_covers
theorem node6619_eq_data : node6619 = Data.node6619 := by
  have branches : (node6607, Graph.empty) = (Data.node6607, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6607_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6619 row children.1 children.2) row6619_eq_data branches
set_option maxRecDepth 4096 in
theorem row6599_eq_data : row6599 = Data.row6599 := rfl
def node6599 : Graph := .node 6599 row6599 node6581 node6619
set_option maxRecDepth 4096 in
theorem node6599_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6599 :=
  Budget.node_covers _ _ 6599 row6599 node6581 node6619
    node6581_covers node6619_covers (by decide) (by decide) row6599_covers
theorem node6599_eq_data : node6599 = Data.node6599 := by
  have branches : (node6581, node6619) = (Data.node6581, Data.node6619) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6581_eq_data node6619_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6599 row children.1 children.2) row6599_eq_data branches
set_option maxRecDepth 4096 in
theorem row6653_eq_data : row6653 = Data.row6653 := rfl
def node6653 : Graph := .node 6653 row6653 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6653_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6653 :=
  Budget.node_covers _ _ 6653 row6653 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6653_covers
theorem node6653_eq_data : node6653 = Data.node6653 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6653 row children.1 children.2) row6653_eq_data branches
set_option maxRecDepth 4096 in
theorem row6659_eq_data : row6659 = Data.row6659 := rfl
def node6659 : Graph := .node 6659 row6659 node6653 Graph.empty
set_option maxRecDepth 4096 in
theorem node6659_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6659 :=
  Budget.node_covers _ _ 6659 row6659 node6653 Graph.empty
    node6653_covers empty_covers (by decide) (by decide) row6659_covers
theorem node6659_eq_data : node6659 = Data.node6659 := by
  have branches : (node6653, Graph.empty) = (Data.node6653, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6653_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6659 row children.1 children.2) row6659_eq_data branches
set_option maxRecDepth 4096 in
theorem row6673_eq_data : row6673 = Data.row6673 := rfl
def node6673 : Graph := .node 6673 row6673 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6673_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6673 :=
  Budget.node_covers _ _ 6673 row6673 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6673_covers
theorem node6673_eq_data : node6673 = Data.node6673 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6673 row children.1 children.2) row6673_eq_data branches
set_option maxRecDepth 4096 in
theorem row6679_eq_data : row6679 = Data.row6679 := rfl
def node6679 : Graph := .node 6679 row6679 node6673 Graph.empty
set_option maxRecDepth 4096 in
theorem node6679_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6679 :=
  Budget.node_covers _ _ 6679 row6679 node6673 Graph.empty
    node6673_covers empty_covers (by decide) (by decide) row6679_covers
theorem node6679_eq_data : node6679 = Data.node6679 := by
  have branches : (node6673, Graph.empty) = (Data.node6673, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6673_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6679 row children.1 children.2) row6679_eq_data branches
set_option maxRecDepth 4096 in
theorem row6661_eq_data : row6661 = Data.row6661 := rfl
def node6661 : Graph := .node 6661 row6661 node6659 node6679
set_option maxRecDepth 4096 in
theorem node6661_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6661 :=
  Budget.node_covers _ _ 6661 row6661 node6659 node6679
    node6659_covers node6679_covers (by decide) (by decide) row6661_covers
theorem node6661_eq_data : node6661 = Data.node6661 := by
  have branches : (node6659, node6679) = (Data.node6659, Data.node6679) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6659_eq_data node6679_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6661 row children.1 children.2) row6661_eq_data branches
set_option maxRecDepth 4096 in
theorem row6637_eq_data : row6637 = Data.row6637 := rfl
def node6637 : Graph := .node 6637 row6637 node6599 node6661
set_option maxRecDepth 4096 in
theorem node6637_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6637 :=
  Budget.node_covers _ _ 6637 row6637 node6599 node6661
    node6599_covers node6661_covers (by decide) (by decide) row6637_covers
theorem node6637_eq_data : node6637 = Data.node6637 := by
  have branches : (node6599, node6661) = (Data.node6599, Data.node6661) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6599_eq_data node6661_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6637 row children.1 children.2) row6637_eq_data branches
set_option maxRecDepth 4096 in
theorem row6691_eq_data : row6691 = Data.row6691 := rfl
def node6691 : Graph := .node 6691 row6691 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6691_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6691 :=
  Budget.node_covers _ _ 6691 row6691 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6691_covers
theorem node6691_eq_data : node6691 = Data.node6691 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6691 row children.1 children.2) row6691_eq_data branches
set_option maxRecDepth 4096 in
theorem row6701_eq_data : row6701 = Data.row6701 := rfl
def node6701 : Graph := .node 6701 row6701 node6691 Graph.empty
set_option maxRecDepth 4096 in
theorem node6701_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6701 :=
  Budget.node_covers _ _ 6701 row6701 node6691 Graph.empty
    node6691_covers empty_covers (by decide) (by decide) row6701_covers
theorem node6701_eq_data : node6701 = Data.node6701 := by
  have branches : (node6691, Graph.empty) = (Data.node6691, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6691_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6701 row children.1 children.2) row6701_eq_data branches
set_option maxRecDepth 4096 in
theorem row6709_eq_data : row6709 = Data.row6709 := rfl
def node6709 : Graph := .node 6709 row6709 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6709_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6709 :=
  Budget.node_covers _ _ 6709 row6709 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6709_covers
theorem node6709_eq_data : node6709 = Data.node6709 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6709 row children.1 children.2) row6709_eq_data branches
set_option maxRecDepth 4096 in
theorem row6719_eq_data : row6719 = Data.row6719 := rfl
def node6719 : Graph := .node 6719 row6719 node6709 Graph.empty
set_option maxRecDepth 4096 in
theorem node6719_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6719 :=
  Budget.node_covers _ _ 6719 row6719 node6709 Graph.empty
    node6709_covers empty_covers (by decide) (by decide) row6719_covers
theorem node6719_eq_data : node6719 = Data.node6719 := by
  have branches : (node6709, Graph.empty) = (Data.node6709, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6709_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6719 row children.1 children.2) row6719_eq_data branches
set_option maxRecDepth 4096 in
theorem row6703_eq_data : row6703 = Data.row6703 := rfl
def node6703 : Graph := .node 6703 row6703 node6701 node6719
set_option maxRecDepth 4096 in
theorem node6703_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6703 :=
  Budget.node_covers _ _ 6703 row6703 node6701 node6719
    node6701_covers node6719_covers (by decide) (by decide) row6703_covers
theorem node6703_eq_data : node6703 = Data.node6703 := by
  have branches : (node6701, node6719) = (Data.node6701, Data.node6719) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6701_eq_data node6719_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6703 row children.1 children.2) row6703_eq_data branches
set_option maxRecDepth 4096 in
theorem row6737_eq_data : row6737 = Data.row6737 := rfl
def node6737 : Graph := .node 6737 row6737 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6737_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6737 :=
  Budget.node_covers _ _ 6737 row6737 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6737_covers
theorem node6737_eq_data : node6737 = Data.node6737 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6737 row children.1 children.2) row6737_eq_data branches
set_option maxRecDepth 4096 in
theorem row6761_eq_data : row6761 = Data.row6761 := rfl
def node6761 : Graph := .node 6761 row6761 node6737 Graph.empty
set_option maxRecDepth 4096 in
theorem node6761_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6761 :=
  Budget.node_covers _ _ 6761 row6761 node6737 Graph.empty
    node6737_covers empty_covers (by decide) (by decide) row6761_covers
theorem node6761_eq_data : node6761 = Data.node6761 := by
  have branches : (node6737, Graph.empty) = (Data.node6737, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6737_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6761 row children.1 children.2) row6761_eq_data branches
set_option maxRecDepth 4096 in
theorem row6779_eq_data : row6779 = Data.row6779 := rfl
def node6779 : Graph := .node 6779 row6779 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6779_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6779 :=
  Budget.node_covers _ _ 6779 row6779 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6779_covers
theorem node6779_eq_data : node6779 = Data.node6779 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6779 row children.1 children.2) row6779_eq_data branches
set_option maxRecDepth 4096 in
theorem row6781_eq_data : row6781 = Data.row6781 := rfl
def node6781 : Graph := .node 6781 row6781 node6779 Graph.empty
set_option maxRecDepth 4096 in
theorem node6781_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6781 :=
  Budget.node_covers _ _ 6781 row6781 node6779 Graph.empty
    node6779_covers empty_covers (by decide) (by decide) row6781_covers
theorem node6781_eq_data : node6781 = Data.node6781 := by
  have branches : (node6779, Graph.empty) = (Data.node6779, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6779_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6781 row children.1 children.2) row6781_eq_data branches
set_option maxRecDepth 4096 in
theorem row6763_eq_data : row6763 = Data.row6763 := rfl
def node6763 : Graph := .node 6763 row6763 node6761 node6781
set_option maxRecDepth 4096 in
theorem node6763_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6763 :=
  Budget.node_covers _ _ 6763 row6763 node6761 node6781
    node6761_covers node6781_covers (by decide) (by decide) row6763_covers
theorem node6763_eq_data : node6763 = Data.node6763 := by
  have branches : (node6761, node6781) = (Data.node6761, Data.node6781) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6761_eq_data node6781_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6763 row children.1 children.2) row6763_eq_data branches
set_option maxRecDepth 4096 in
theorem row6733_eq_data : row6733 = Data.row6733 := rfl
def node6733 : Graph := .node 6733 row6733 node6703 node6763
set_option maxRecDepth 4096 in
theorem node6733_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6733 :=
  Budget.node_covers _ _ 6733 row6733 node6703 node6763
    node6703_covers node6763_covers (by decide) (by decide) row6733_covers
theorem node6733_eq_data : node6733 = Data.node6733 := by
  have branches : (node6703, node6763) = (Data.node6703, Data.node6763) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6703_eq_data node6763_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6733 row children.1 children.2) row6733_eq_data branches
set_option maxRecDepth 4096 in
theorem row6689_eq_data : row6689 = Data.row6689 := rfl
def node6689 : Graph := .node 6689 row6689 node6637 node6733
set_option maxRecDepth 4096 in
theorem node6689_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6689 :=
  Budget.node_covers _ _ 6689 row6689 node6637 node6733
    node6637_covers node6733_covers (by decide) (by decide) row6689_covers
theorem node6689_eq_data : node6689 = Data.node6689 := by
  have branches : (node6637, node6733) = (Data.node6637, Data.node6733) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6637_eq_data node6733_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6689 row children.1 children.2) row6689_eq_data branches
set_option maxRecDepth 4096 in
theorem row6571_eq_data : row6571 = Data.row6571 := rfl
def node6571 : Graph := .node 6571 row6571 node6451 node6689
set_option maxRecDepth 4096 in
theorem node6571_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6571 :=
  Budget.node_covers _ _ 6571 row6571 node6451 node6689
    node6451_covers node6689_covers (by decide) (by decide) row6571_covers
theorem node6571_eq_data : node6571 = Data.node6571 := by
  have branches : (node6451, node6689) = (Data.node6451, Data.node6689) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6451_eq_data node6689_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6571 row children.1 children.2) row6571_eq_data branches
set_option maxRecDepth 4096 in
theorem row6343_eq_data : row6343 = Data.row6343 := rfl
def node6343 : Graph := .node 6343 row6343 node6151 node6571
set_option maxRecDepth 4096 in
theorem node6343_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6343 :=
  Budget.node_covers _ _ 6343 row6343 node6151 node6571
    node6151_covers node6571_covers (by decide) (by decide) row6343_covers
theorem node6343_eq_data : node6343 = Data.node6343 := by
  have branches : (node6151, node6571) = (Data.node6151, Data.node6571) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6151_eq_data node6571_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6343 row children.1 children.2) row6343_eq_data branches
set_option maxRecDepth 4096 in
theorem row6793_eq_data : row6793 = Data.row6793 := rfl
def node6793 : Graph := .node 6793 row6793 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6793_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6793 :=
  Budget.node_covers _ _ 6793 row6793 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6793_covers
theorem node6793_eq_data : node6793 = Data.node6793 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6793 row children.1 children.2) row6793_eq_data branches
set_option maxRecDepth 4096 in
theorem row6823_eq_data : row6823 = Data.row6823 := rfl
def node6823 : Graph := .node 6823 row6823 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6823_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6823 :=
  Budget.node_covers _ _ 6823 row6823 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6823_covers
theorem node6823_eq_data : node6823 = Data.node6823 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6823 row children.1 children.2) row6823_eq_data branches
set_option maxRecDepth 4096 in
theorem row6803_eq_data : row6803 = Data.row6803 := rfl
def node6803 : Graph := .node 6803 row6803 node6793 node6823
set_option maxRecDepth 4096 in
theorem node6803_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6803 :=
  Budget.node_covers _ _ 6803 row6803 node6793 node6823
    node6793_covers node6823_covers (by decide) (by decide) row6803_covers
theorem node6803_eq_data : node6803 = Data.node6803 := by
  have branches : (node6793, node6823) = (Data.node6793, Data.node6823) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6793_eq_data node6823_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6803 row children.1 children.2) row6803_eq_data branches
set_option maxRecDepth 4096 in
theorem row6829_eq_data : row6829 = Data.row6829 := rfl
def node6829 : Graph := .node 6829 row6829 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6829_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6829 :=
  Budget.node_covers _ _ 6829 row6829 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6829_covers
theorem node6829_eq_data : node6829 = Data.node6829 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6829 row children.1 children.2) row6829_eq_data branches
set_option maxRecDepth 4096 in
theorem row6833_eq_data : row6833 = Data.row6833 := rfl
def node6833 : Graph := .node 6833 row6833 node6829 Graph.empty
set_option maxRecDepth 4096 in
theorem node6833_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6833 :=
  Budget.node_covers _ _ 6833 row6833 node6829 Graph.empty
    node6829_covers empty_covers (by decide) (by decide) row6833_covers
theorem node6833_eq_data : node6833 = Data.node6833 := by
  have branches : (node6829, Graph.empty) = (Data.node6829, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6829_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6833 row children.1 children.2) row6833_eq_data branches
set_option maxRecDepth 4096 in
theorem row6827_eq_data : row6827 = Data.row6827 := rfl
def node6827 : Graph := .node 6827 row6827 node6803 node6833
set_option maxRecDepth 4096 in
theorem node6827_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6827 :=
  Budget.node_covers _ _ 6827 row6827 node6803 node6833
    node6803_covers node6833_covers (by decide) (by decide) row6827_covers
theorem node6827_eq_data : node6827 = Data.node6827 := by
  have branches : (node6803, node6833) = (Data.node6803, Data.node6833) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6803_eq_data node6833_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6827 row children.1 children.2) row6827_eq_data branches
set_option maxRecDepth 4096 in
theorem row6857_eq_data : row6857 = Data.row6857 := rfl
def node6857 : Graph := .node 6857 row6857 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6857_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6857 :=
  Budget.node_covers _ _ 6857 row6857 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6857_covers
theorem node6857_eq_data : node6857 = Data.node6857 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6857 row children.1 children.2) row6857_eq_data branches
set_option maxRecDepth 4096 in
theorem row6863_eq_data : row6863 = Data.row6863 := rfl
def node6863 : Graph := .node 6863 row6863 node6857 Graph.empty
set_option maxRecDepth 4096 in
theorem node6863_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6863 :=
  Budget.node_covers _ _ 6863 row6863 node6857 Graph.empty
    node6857_covers empty_covers (by decide) (by decide) row6863_covers
theorem node6863_eq_data : node6863 = Data.node6863 := by
  have branches : (node6857, Graph.empty) = (Data.node6857, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6857_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6863 row children.1 children.2) row6863_eq_data branches
set_option maxRecDepth 4096 in
theorem row6871_eq_data : row6871 = Data.row6871 := rfl
def node6871 : Graph := .node 6871 row6871 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6871_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6871 :=
  Budget.node_covers _ _ 6871 row6871 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6871_covers
theorem node6871_eq_data : node6871 = Data.node6871 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6871 row children.1 children.2) row6871_eq_data branches
set_option maxRecDepth 4096 in
theorem row6883_eq_data : row6883 = Data.row6883 := rfl
def node6883 : Graph := .node 6883 row6883 node6871 Graph.empty
set_option maxRecDepth 4096 in
theorem node6883_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6883 :=
  Budget.node_covers _ _ 6883 row6883 node6871 Graph.empty
    node6871_covers empty_covers (by decide) (by decide) row6883_covers
theorem node6883_eq_data : node6883 = Data.node6883 := by
  have branches : (node6871, Graph.empty) = (Data.node6871, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6871_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6883 row children.1 children.2) row6883_eq_data branches
set_option maxRecDepth 4096 in
theorem row6869_eq_data : row6869 = Data.row6869 := rfl
def node6869 : Graph := .node 6869 row6869 node6863 node6883
set_option maxRecDepth 4096 in
theorem node6869_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6869 :=
  Budget.node_covers _ _ 6869 row6869 node6863 node6883
    node6863_covers node6883_covers (by decide) (by decide) row6869_covers
theorem node6869_eq_data : node6869 = Data.node6869 := by
  have branches : (node6863, node6883) = (Data.node6863, Data.node6883) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6863_eq_data node6883_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6869 row children.1 children.2) row6869_eq_data branches
set_option maxRecDepth 4096 in
theorem row6841_eq_data : row6841 = Data.row6841 := rfl
def node6841 : Graph := .node 6841 row6841 node6827 node6869
set_option maxRecDepth 4096 in
theorem node6841_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6841 :=
  Budget.node_covers _ _ 6841 row6841 node6827 node6869
    node6827_covers node6869_covers (by decide) (by decide) row6841_covers
theorem node6841_eq_data : node6841 = Data.node6841 := by
  have branches : (node6827, node6869) = (Data.node6827, Data.node6869) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6827_eq_data node6869_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6841 row children.1 children.2) row6841_eq_data branches
set_option maxRecDepth 4096 in
theorem row6907_eq_data : row6907 = Data.row6907 := rfl
def node6907 : Graph := .node 6907 row6907 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6907_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6907 :=
  Budget.node_covers _ _ 6907 row6907 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6907_covers
theorem node6907_eq_data : node6907 = Data.node6907 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6907 row children.1 children.2) row6907_eq_data branches
set_option maxRecDepth 4096 in
theorem row6911_eq_data : row6911 = Data.row6911 := rfl
def node6911 : Graph := .node 6911 row6911 node6907 Graph.empty
set_option maxRecDepth 4096 in
theorem node6911_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6911 :=
  Budget.node_covers _ _ 6911 row6911 node6907 Graph.empty
    node6907_covers empty_covers (by decide) (by decide) row6911_covers
theorem node6911_eq_data : node6911 = Data.node6911 := by
  have branches : (node6907, Graph.empty) = (Data.node6907, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6907_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6911 row children.1 children.2) row6911_eq_data branches
set_option maxRecDepth 4096 in
theorem row6947_eq_data : row6947 = Data.row6947 := rfl
def node6947 : Graph := .node 6947 row6947 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6947_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6947 :=
  Budget.node_covers _ _ 6947 row6947 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6947_covers
theorem node6947_eq_data : node6947 = Data.node6947 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6947 row children.1 children.2) row6947_eq_data branches
set_option maxRecDepth 4096 in
theorem row6949_eq_data : row6949 = Data.row6949 := rfl
def node6949 : Graph := .node 6949 row6949 node6947 Graph.empty
set_option maxRecDepth 4096 in
theorem node6949_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6949 :=
  Budget.node_covers _ _ 6949 row6949 node6947 Graph.empty
    node6947_covers empty_covers (by decide) (by decide) row6949_covers
theorem node6949_eq_data : node6949 = Data.node6949 := by
  have branches : (node6947, Graph.empty) = (Data.node6947, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6947_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6949 row children.1 children.2) row6949_eq_data branches
set_option maxRecDepth 4096 in
theorem row6917_eq_data : row6917 = Data.row6917 := rfl
def node6917 : Graph := .node 6917 row6917 node6911 node6949
set_option maxRecDepth 4096 in
theorem node6917_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6917 :=
  Budget.node_covers _ _ 6917 row6917 node6911 node6949
    node6911_covers node6949_covers (by decide) (by decide) row6917_covers
theorem node6917_eq_data : node6917 = Data.node6917 := by
  have branches : (node6911, node6949) = (Data.node6911, Data.node6949) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6911_eq_data node6949_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6917 row children.1 children.2) row6917_eq_data branches
set_option maxRecDepth 4096 in
theorem row6961_eq_data : row6961 = Data.row6961 := rfl
def node6961 : Graph := .node 6961 row6961 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6961_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6961 :=
  Budget.node_covers _ _ 6961 row6961 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6961_covers
theorem node6961_eq_data : node6961 = Data.node6961 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6961 row children.1 children.2) row6961_eq_data branches
set_option maxRecDepth 4096 in
theorem row6967_eq_data : row6967 = Data.row6967 := rfl
def node6967 : Graph := .node 6967 row6967 node6961 Graph.empty
set_option maxRecDepth 4096 in
theorem node6967_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6967 :=
  Budget.node_covers _ _ 6967 row6967 node6961 Graph.empty
    node6961_covers empty_covers (by decide) (by decide) row6967_covers
theorem node6967_eq_data : node6967 = Data.node6967 := by
  have branches : (node6961, Graph.empty) = (Data.node6961, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6961_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6967 row children.1 children.2) row6967_eq_data branches
set_option maxRecDepth 4096 in
theorem row6977_eq_data : row6977 = Data.row6977 := rfl
def node6977 : Graph := .node 6977 row6977 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6977_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6977 :=
  Budget.node_covers _ _ 6977 row6977 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6977_covers
theorem node6977_eq_data : node6977 = Data.node6977 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6977 row children.1 children.2) row6977_eq_data branches
set_option maxRecDepth 4096 in
theorem row6983_eq_data : row6983 = Data.row6983 := rfl
def node6983 : Graph := .node 6983 row6983 node6977 Graph.empty
set_option maxRecDepth 4096 in
theorem node6983_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6983 :=
  Budget.node_covers _ _ 6983 row6983 node6977 Graph.empty
    node6977_covers empty_covers (by decide) (by decide) row6983_covers
theorem node6983_eq_data : node6983 = Data.node6983 := by
  have branches : (node6977, Graph.empty) = (Data.node6977, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6977_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6983 row children.1 children.2) row6983_eq_data branches
set_option maxRecDepth 4096 in
theorem row6971_eq_data : row6971 = Data.row6971 := rfl
def node6971 : Graph := .node 6971 row6971 node6967 node6983
set_option maxRecDepth 4096 in
theorem node6971_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6971 :=
  Budget.node_covers _ _ 6971 row6971 node6967 node6983
    node6967_covers node6983_covers (by decide) (by decide) row6971_covers
theorem node6971_eq_data : node6971 = Data.node6971 := by
  have branches : (node6967, node6983) = (Data.node6967, Data.node6983) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6967_eq_data node6983_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6971 row children.1 children.2) row6971_eq_data branches
set_option maxRecDepth 4096 in
theorem row6959_eq_data : row6959 = Data.row6959 := rfl
def node6959 : Graph := .node 6959 row6959 node6917 node6971
set_option maxRecDepth 4096 in
theorem node6959_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6959 :=
  Budget.node_covers _ _ 6959 row6959 node6917 node6971
    node6917_covers node6971_covers (by decide) (by decide) row6959_covers
theorem node6959_eq_data : node6959 = Data.node6959 := by
  have branches : (node6917, node6971) = (Data.node6917, Data.node6971) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6917_eq_data node6971_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6959 row children.1 children.2) row6959_eq_data branches
set_option maxRecDepth 4096 in
theorem row6899_eq_data : row6899 = Data.row6899 := rfl
def node6899 : Graph := .node 6899 row6899 node6841 node6959
set_option maxRecDepth 4096 in
theorem node6899_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6899 :=
  Budget.node_covers _ _ 6899 row6899 node6841 node6959
    node6841_covers node6959_covers (by decide) (by decide) row6899_covers
theorem node6899_eq_data : node6899 = Data.node6899 := by
  have branches : (node6841, node6959) = (Data.node6841, Data.node6959) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6841_eq_data node6959_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6899 row children.1 children.2) row6899_eq_data branches
set_option maxRecDepth 4096 in
theorem row6997_eq_data : row6997 = Data.row6997 := rfl
def node6997 : Graph := .node 6997 row6997 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node6997_covers : Budget.Covers PrimePairVertices.largeTree 26034 node6997 :=
  Budget.node_covers _ _ 6997 row6997 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row6997_covers
theorem node6997_eq_data : node6997 = Data.node6997 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 6997 row children.1 children.2) row6997_eq_data branches
set_option maxRecDepth 4096 in
theorem row7001_eq_data : row7001 = Data.row7001 := rfl
def node7001 : Graph := .node 7001 row7001 node6997 Graph.empty
set_option maxRecDepth 4096 in
theorem node7001_covers : Budget.Covers PrimePairVertices.largeTree 26034 node7001 :=
  Budget.node_covers _ _ 7001 row7001 node6997 Graph.empty
    node6997_covers empty_covers (by decide) (by decide) row7001_covers
theorem node7001_eq_data : node7001 = Data.node7001 := by
  have branches : (node6997, Graph.empty) = (Data.node6997, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node6997_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 7001 row children.1 children.2) row7001_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
