import ProjectEulerSolutions.Lib.PrimePairLarge.Graph019
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row10427_eq_data : row10427 = Data.row10427 := rfl
def node10427 : Graph := .node 10427 row10427 node10337 node10463
set_option maxRecDepth 4096 in
theorem node10427_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10427 :=
  Budget.node_covers _ _ 10427 row10427 node10337 node10463
    node10337_covers node10463_covers (by decide) (by decide) row10427_covers
theorem node10427_eq_data : node10427 = Data.node10427 := by
  have branches : (node10337, node10463) = (Data.node10337, Data.node10463) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10337_eq_data node10463_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10427 row children.1 children.2) row10427_eq_data branches
set_option maxRecDepth 4096 in
theorem row10531_eq_data : row10531 = Data.row10531 := rfl
def node10531 : Graph := .node 10531 row10531 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10531_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10531 :=
  Budget.node_covers _ _ 10531 row10531 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10531_covers
theorem node10531_eq_data : node10531 = Data.node10531 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10531 row children.1 children.2) row10531_eq_data branches
set_option maxRecDepth 4096 in
theorem row10559_eq_data : row10559 = Data.row10559 := rfl
def node10559 : Graph := .node 10559 row10559 node10531 Graph.empty
set_option maxRecDepth 4096 in
theorem node10559_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10559 :=
  Budget.node_covers _ _ 10559 row10559 node10531 Graph.empty
    node10531_covers empty_covers (by decide) (by decide) row10559_covers
theorem node10559_eq_data : node10559 = Data.node10559 := by
  have branches : (node10531, Graph.empty) = (Data.node10531, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10531_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10559 row children.1 children.2) row10559_eq_data branches
set_option maxRecDepth 4096 in
theorem row10589_eq_data : row10589 = Data.row10589 := rfl
def node10589 : Graph := .node 10589 row10589 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10589_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10589 :=
  Budget.node_covers _ _ 10589 row10589 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10589_covers
theorem node10589_eq_data : node10589 = Data.node10589 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10589 row children.1 children.2) row10589_eq_data branches
set_option maxRecDepth 4096 in
theorem row10597_eq_data : row10597 = Data.row10597 := rfl
def node10597 : Graph := .node 10597 row10597 node10589 Graph.empty
set_option maxRecDepth 4096 in
theorem node10597_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10597 :=
  Budget.node_covers _ _ 10597 row10597 node10589 Graph.empty
    node10589_covers empty_covers (by decide) (by decide) row10597_covers
theorem node10597_eq_data : node10597 = Data.node10597 := by
  have branches : (node10589, Graph.empty) = (Data.node10589, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10589_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10597 row children.1 children.2) row10597_eq_data branches
set_option maxRecDepth 4096 in
theorem row10567_eq_data : row10567 = Data.row10567 := rfl
def node10567 : Graph := .node 10567 row10567 node10559 node10597
set_option maxRecDepth 4096 in
theorem node10567_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10567 :=
  Budget.node_covers _ _ 10567 row10567 node10559 node10597
    node10559_covers node10597_covers (by decide) (by decide) row10567_covers
theorem node10567_eq_data : node10567 = Data.node10567 := by
  have branches : (node10559, node10597) = (Data.node10559, Data.node10597) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10559_eq_data node10597_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10567 row children.1 children.2) row10567_eq_data branches
set_option maxRecDepth 4096 in
theorem row10607_eq_data : row10607 = Data.row10607 := rfl
def node10607 : Graph := .node 10607 row10607 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10607_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10607 :=
  Budget.node_covers _ _ 10607 row10607 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10607_covers
theorem node10607_eq_data : node10607 = Data.node10607 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10607 row children.1 children.2) row10607_eq_data branches
set_option maxRecDepth 4096 in
theorem row10613_eq_data : row10613 = Data.row10613 := rfl
def node10613 : Graph := .node 10613 row10613 node10607 Graph.empty
set_option maxRecDepth 4096 in
theorem node10613_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10613 :=
  Budget.node_covers _ _ 10613 row10613 node10607 Graph.empty
    node10607_covers empty_covers (by decide) (by decide) row10613_covers
theorem node10613_eq_data : node10613 = Data.node10613 := by
  have branches : (node10607, Graph.empty) = (Data.node10607, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10607_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10613 row children.1 children.2) row10613_eq_data branches
set_option maxRecDepth 4096 in
theorem row10631_eq_data : row10631 = Data.row10631 := rfl
def node10631 : Graph := .node 10631 row10631 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10631_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10631 :=
  Budget.node_covers _ _ 10631 row10631 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10631_covers
theorem node10631_eq_data : node10631 = Data.node10631 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10631 row children.1 children.2) row10631_eq_data branches
set_option maxRecDepth 4096 in
theorem row10639_eq_data : row10639 = Data.row10639 := rfl
def node10639 : Graph := .node 10639 row10639 node10631 Graph.empty
set_option maxRecDepth 4096 in
theorem node10639_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10639 :=
  Budget.node_covers _ _ 10639 row10639 node10631 Graph.empty
    node10631_covers empty_covers (by decide) (by decide) row10639_covers
theorem node10639_eq_data : node10639 = Data.node10639 := by
  have branches : (node10631, Graph.empty) = (Data.node10631, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10631_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10639 row children.1 children.2) row10639_eq_data branches
set_option maxRecDepth 4096 in
theorem row10627_eq_data : row10627 = Data.row10627 := rfl
def node10627 : Graph := .node 10627 row10627 node10613 node10639
set_option maxRecDepth 4096 in
theorem node10627_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10627 :=
  Budget.node_covers _ _ 10627 row10627 node10613 node10639
    node10613_covers node10639_covers (by decide) (by decide) row10627_covers
theorem node10627_eq_data : node10627 = Data.node10627 := by
  have branches : (node10613, node10639) = (Data.node10613, Data.node10639) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10613_eq_data node10639_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10627 row children.1 children.2) row10627_eq_data branches
set_option maxRecDepth 4096 in
theorem row10601_eq_data : row10601 = Data.row10601 := rfl
def node10601 : Graph := .node 10601 row10601 node10567 node10627
set_option maxRecDepth 4096 in
theorem node10601_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10601 :=
  Budget.node_covers _ _ 10601 row10601 node10567 node10627
    node10567_covers node10627_covers (by decide) (by decide) row10601_covers
theorem node10601_eq_data : node10601 = Data.node10601 := by
  have branches : (node10567, node10627) = (Data.node10567, Data.node10627) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10567_eq_data node10627_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10601 row children.1 children.2) row10601_eq_data branches
set_option maxRecDepth 4096 in
theorem row10657_eq_data : row10657 = Data.row10657 := rfl
def node10657 : Graph := .node 10657 row10657 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10657_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10657 :=
  Budget.node_covers _ _ 10657 row10657 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10657_covers
theorem node10657_eq_data : node10657 = Data.node10657 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10657 row children.1 children.2) row10657_eq_data branches
set_option maxRecDepth 4096 in
theorem row10663_eq_data : row10663 = Data.row10663 := rfl
def node10663 : Graph := .node 10663 row10663 node10657 Graph.empty
set_option maxRecDepth 4096 in
theorem node10663_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10663 :=
  Budget.node_covers _ _ 10663 row10663 node10657 Graph.empty
    node10657_covers empty_covers (by decide) (by decide) row10663_covers
theorem node10663_eq_data : node10663 = Data.node10663 := by
  have branches : (node10657, Graph.empty) = (Data.node10657, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10657_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10663 row children.1 children.2) row10663_eq_data branches
set_option maxRecDepth 4096 in
theorem row10687_eq_data : row10687 = Data.row10687 := rfl
def node10687 : Graph := .node 10687 row10687 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10687_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10687 :=
  Budget.node_covers _ _ 10687 row10687 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10687_covers
theorem node10687_eq_data : node10687 = Data.node10687 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10687 row children.1 children.2) row10687_eq_data branches
set_option maxRecDepth 4096 in
theorem row10691_eq_data : row10691 = Data.row10691 := rfl
def node10691 : Graph := .node 10691 row10691 node10687 Graph.empty
set_option maxRecDepth 4096 in
theorem node10691_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10691 :=
  Budget.node_covers _ _ 10691 row10691 node10687 Graph.empty
    node10687_covers empty_covers (by decide) (by decide) row10691_covers
theorem node10691_eq_data : node10691 = Data.node10691 := by
  have branches : (node10687, Graph.empty) = (Data.node10687, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10687_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10691 row children.1 children.2) row10691_eq_data branches
set_option maxRecDepth 4096 in
theorem row10667_eq_data : row10667 = Data.row10667 := rfl
def node10667 : Graph := .node 10667 row10667 node10663 node10691
set_option maxRecDepth 4096 in
theorem node10667_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10667 :=
  Budget.node_covers _ _ 10667 row10667 node10663 node10691
    node10663_covers node10691_covers (by decide) (by decide) row10667_covers
theorem node10667_eq_data : node10667 = Data.node10667 := by
  have branches : (node10663, node10691) = (Data.node10663, Data.node10691) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10663_eq_data node10691_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10667 row children.1 children.2) row10667_eq_data branches
set_option maxRecDepth 4096 in
theorem row10711_eq_data : row10711 = Data.row10711 := rfl
def node10711 : Graph := .node 10711 row10711 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10711_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10711 :=
  Budget.node_covers _ _ 10711 row10711 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10711_covers
theorem node10711_eq_data : node10711 = Data.node10711 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10711 row children.1 children.2) row10711_eq_data branches
set_option maxRecDepth 4096 in
theorem row10723_eq_data : row10723 = Data.row10723 := rfl
def node10723 : Graph := .node 10723 row10723 node10711 Graph.empty
set_option maxRecDepth 4096 in
theorem node10723_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10723 :=
  Budget.node_covers _ _ 10723 row10723 node10711 Graph.empty
    node10711_covers empty_covers (by decide) (by decide) row10723_covers
theorem node10723_eq_data : node10723 = Data.node10723 := by
  have branches : (node10711, Graph.empty) = (Data.node10711, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10711_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10723 row children.1 children.2) row10723_eq_data branches
set_option maxRecDepth 4096 in
theorem row10733_eq_data : row10733 = Data.row10733 := rfl
def node10733 : Graph := .node 10733 row10733 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10733_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10733 :=
  Budget.node_covers _ _ 10733 row10733 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10733_covers
theorem node10733_eq_data : node10733 = Data.node10733 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10733 row children.1 children.2) row10733_eq_data branches
set_option maxRecDepth 4096 in
theorem row10739_eq_data : row10739 = Data.row10739 := rfl
def node10739 : Graph := .node 10739 row10739 node10733 Graph.empty
set_option maxRecDepth 4096 in
theorem node10739_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10739 :=
  Budget.node_covers _ _ 10739 row10739 node10733 Graph.empty
    node10733_covers empty_covers (by decide) (by decide) row10739_covers
theorem node10739_eq_data : node10739 = Data.node10739 := by
  have branches : (node10733, Graph.empty) = (Data.node10733, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10733_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10739 row children.1 children.2) row10739_eq_data branches
set_option maxRecDepth 4096 in
theorem row10729_eq_data : row10729 = Data.row10729 := rfl
def node10729 : Graph := .node 10729 row10729 node10723 node10739
set_option maxRecDepth 4096 in
theorem node10729_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10729 :=
  Budget.node_covers _ _ 10729 row10729 node10723 node10739
    node10723_covers node10739_covers (by decide) (by decide) row10729_covers
theorem node10729_eq_data : node10729 = Data.node10729 := by
  have branches : (node10723, node10739) = (Data.node10723, Data.node10739) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10723_eq_data node10739_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10729 row children.1 children.2) row10729_eq_data branches
set_option maxRecDepth 4096 in
theorem row10709_eq_data : row10709 = Data.row10709 := rfl
def node10709 : Graph := .node 10709 row10709 node10667 node10729
set_option maxRecDepth 4096 in
theorem node10709_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10709 :=
  Budget.node_covers _ _ 10709 row10709 node10667 node10729
    node10667_covers node10729_covers (by decide) (by decide) row10709_covers
theorem node10709_eq_data : node10709 = Data.node10709 := by
  have branches : (node10667, node10729) = (Data.node10667, Data.node10729) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10667_eq_data node10729_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10709 row children.1 children.2) row10709_eq_data branches
set_option maxRecDepth 4096 in
theorem row10651_eq_data : row10651 = Data.row10651 := rfl
def node10651 : Graph := .node 10651 row10651 node10601 node10709
set_option maxRecDepth 4096 in
theorem node10651_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10651 :=
  Budget.node_covers _ _ 10651 row10651 node10601 node10709
    node10601_covers node10709_covers (by decide) (by decide) row10651_covers
theorem node10651_eq_data : node10651 = Data.node10651 := by
  have branches : (node10601, node10709) = (Data.node10601, Data.node10709) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10601_eq_data node10709_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10651 row children.1 children.2) row10651_eq_data branches
set_option maxRecDepth 4096 in
theorem row10529_eq_data : row10529 = Data.row10529 := rfl
def node10529 : Graph := .node 10529 row10529 node10427 node10651
set_option maxRecDepth 4096 in
theorem node10529_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10529 :=
  Budget.node_covers _ _ 10529 row10529 node10427 node10651
    node10427_covers node10651_covers (by decide) (by decide) row10529_covers
theorem node10529_eq_data : node10529 = Data.node10529 := by
  have branches : (node10427, node10651) = (Data.node10427, Data.node10651) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10427_eq_data node10651_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10529 row children.1 children.2) row10529_eq_data branches
set_option maxRecDepth 4096 in
theorem row10771_eq_data : row10771 = Data.row10771 := rfl
def node10771 : Graph := .node 10771 row10771 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10771_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10771 :=
  Budget.node_covers _ _ 10771 row10771 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10771_covers
theorem node10771_eq_data : node10771 = Data.node10771 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10771 row children.1 children.2) row10771_eq_data branches
set_option maxRecDepth 4096 in
theorem row10781_eq_data : row10781 = Data.row10781 := rfl
def node10781 : Graph := .node 10781 row10781 node10771 Graph.empty
set_option maxRecDepth 4096 in
theorem node10781_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10781 :=
  Budget.node_covers _ _ 10781 row10781 node10771 Graph.empty
    node10771_covers empty_covers (by decide) (by decide) row10781_covers
theorem node10781_eq_data : node10781 = Data.node10781 := by
  have branches : (node10771, Graph.empty) = (Data.node10771, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10771_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10781 row children.1 children.2) row10781_eq_data branches
set_option maxRecDepth 4096 in
theorem row10799_eq_data : row10799 = Data.row10799 := rfl
def node10799 : Graph := .node 10799 row10799 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10799_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10799 :=
  Budget.node_covers _ _ 10799 row10799 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10799_covers
theorem node10799_eq_data : node10799 = Data.node10799 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10799 row children.1 children.2) row10799_eq_data branches
set_option maxRecDepth 4096 in
theorem row10831_eq_data : row10831 = Data.row10831 := rfl
def node10831 : Graph := .node 10831 row10831 node10799 Graph.empty
set_option maxRecDepth 4096 in
theorem node10831_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10831 :=
  Budget.node_covers _ _ 10831 row10831 node10799 Graph.empty
    node10799_covers empty_covers (by decide) (by decide) row10831_covers
theorem node10831_eq_data : node10831 = Data.node10831 := by
  have branches : (node10799, Graph.empty) = (Data.node10799, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10799_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10831 row children.1 children.2) row10831_eq_data branches
set_option maxRecDepth 4096 in
theorem row10789_eq_data : row10789 = Data.row10789 := rfl
def node10789 : Graph := .node 10789 row10789 node10781 node10831
set_option maxRecDepth 4096 in
theorem node10789_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10789 :=
  Budget.node_covers _ _ 10789 row10789 node10781 node10831
    node10781_covers node10831_covers (by decide) (by decide) row10789_covers
theorem node10789_eq_data : node10789 = Data.node10789 := by
  have branches : (node10781, node10831) = (Data.node10781, Data.node10831) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10781_eq_data node10831_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10789 row children.1 children.2) row10789_eq_data branches
set_option maxRecDepth 4096 in
theorem row10847_eq_data : row10847 = Data.row10847 := rfl
def node10847 : Graph := .node 10847 row10847 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10847_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10847 :=
  Budget.node_covers _ _ 10847 row10847 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10847_covers
theorem node10847_eq_data : node10847 = Data.node10847 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10847 row children.1 children.2) row10847_eq_data branches
set_option maxRecDepth 4096 in
theorem row10853_eq_data : row10853 = Data.row10853 := rfl
def node10853 : Graph := .node 10853 row10853 node10847 Graph.empty
set_option maxRecDepth 4096 in
theorem node10853_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10853 :=
  Budget.node_covers _ _ 10853 row10853 node10847 Graph.empty
    node10847_covers empty_covers (by decide) (by decide) row10853_covers
theorem node10853_eq_data : node10853 = Data.node10853 := by
  have branches : (node10847, Graph.empty) = (Data.node10847, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10847_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10853 row children.1 children.2) row10853_eq_data branches
set_option maxRecDepth 4096 in
theorem row10861_eq_data : row10861 = Data.row10861 := rfl
def node10861 : Graph := .node 10861 row10861 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10861_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10861 :=
  Budget.node_covers _ _ 10861 row10861 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10861_covers
theorem node10861_eq_data : node10861 = Data.node10861 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10861 row children.1 children.2) row10861_eq_data branches
set_option maxRecDepth 4096 in
theorem row10867_eq_data : row10867 = Data.row10867 := rfl
def node10867 : Graph := .node 10867 row10867 node10861 Graph.empty
set_option maxRecDepth 4096 in
theorem node10867_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10867 :=
  Budget.node_covers _ _ 10867 row10867 node10861 Graph.empty
    node10861_covers empty_covers (by decide) (by decide) row10867_covers
theorem node10867_eq_data : node10867 = Data.node10867 := by
  have branches : (node10861, Graph.empty) = (Data.node10861, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10861_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10867 row children.1 children.2) row10867_eq_data branches
set_option maxRecDepth 4096 in
theorem row10859_eq_data : row10859 = Data.row10859 := rfl
def node10859 : Graph := .node 10859 row10859 node10853 node10867
set_option maxRecDepth 4096 in
theorem node10859_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10859 :=
  Budget.node_covers _ _ 10859 row10859 node10853 node10867
    node10853_covers node10867_covers (by decide) (by decide) row10859_covers
theorem node10859_eq_data : node10859 = Data.node10859 := by
  have branches : (node10853, node10867) = (Data.node10853, Data.node10867) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10853_eq_data node10867_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10859 row children.1 children.2) row10859_eq_data branches
set_option maxRecDepth 4096 in
theorem row10837_eq_data : row10837 = Data.row10837 := rfl
def node10837 : Graph := .node 10837 row10837 node10789 node10859
set_option maxRecDepth 4096 in
theorem node10837_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10837 :=
  Budget.node_covers _ _ 10837 row10837 node10789 node10859
    node10789_covers node10859_covers (by decide) (by decide) row10837_covers
theorem node10837_eq_data : node10837 = Data.node10837 := by
  have branches : (node10789, node10859) = (Data.node10789, Data.node10859) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10789_eq_data node10859_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10837 row children.1 children.2) row10837_eq_data branches
set_option maxRecDepth 4096 in
theorem row10889_eq_data : row10889 = Data.row10889 := rfl
def node10889 : Graph := .node 10889 row10889 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10889_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10889 :=
  Budget.node_covers _ _ 10889 row10889 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10889_covers
theorem node10889_eq_data : node10889 = Data.node10889 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10889 row children.1 children.2) row10889_eq_data branches
set_option maxRecDepth 4096 in
theorem row10891_eq_data : row10891 = Data.row10891 := rfl
def node10891 : Graph := .node 10891 row10891 node10889 Graph.empty
set_option maxRecDepth 4096 in
theorem node10891_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10891 :=
  Budget.node_covers _ _ 10891 row10891 node10889 Graph.empty
    node10889_covers empty_covers (by decide) (by decide) row10891_covers
theorem node10891_eq_data : node10891 = Data.node10891 := by
  have branches : (node10889, Graph.empty) = (Data.node10889, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10889_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10891 row children.1 children.2) row10891_eq_data branches
set_option maxRecDepth 4096 in
theorem row10909_eq_data : row10909 = Data.row10909 := rfl
def node10909 : Graph := .node 10909 row10909 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10909_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10909 :=
  Budget.node_covers _ _ 10909 row10909 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10909_covers
theorem node10909_eq_data : node10909 = Data.node10909 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10909 row children.1 children.2) row10909_eq_data branches
set_option maxRecDepth 4096 in
theorem row10937_eq_data : row10937 = Data.row10937 := rfl
def node10937 : Graph := .node 10937 row10937 node10909 Graph.empty
set_option maxRecDepth 4096 in
theorem node10937_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10937 :=
  Budget.node_covers _ _ 10937 row10937 node10909 Graph.empty
    node10909_covers empty_covers (by decide) (by decide) row10937_covers
theorem node10937_eq_data : node10937 = Data.node10937 := by
  have branches : (node10909, Graph.empty) = (Data.node10909, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10909_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10937 row children.1 children.2) row10937_eq_data branches
set_option maxRecDepth 4096 in
theorem row10903_eq_data : row10903 = Data.row10903 := rfl
def node10903 : Graph := .node 10903 row10903 node10891 node10937
set_option maxRecDepth 4096 in
theorem node10903_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10903 :=
  Budget.node_covers _ _ 10903 row10903 node10891 node10937
    node10891_covers node10937_covers (by decide) (by decide) row10903_covers
theorem node10903_eq_data : node10903 = Data.node10903 := by
  have branches : (node10891, node10937) = (Data.node10891, Data.node10937) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10891_eq_data node10937_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10903 row children.1 children.2) row10903_eq_data branches
set_option maxRecDepth 4096 in
theorem row10949_eq_data : row10949 = Data.row10949 := rfl
def node10949 : Graph := .node 10949 row10949 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10949_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10949 :=
  Budget.node_covers _ _ 10949 row10949 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10949_covers
theorem node10949_eq_data : node10949 = Data.node10949 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10949 row children.1 children.2) row10949_eq_data branches
set_option maxRecDepth 4096 in
theorem row10957_eq_data : row10957 = Data.row10957 := rfl
def node10957 : Graph := .node 10957 row10957 node10949 Graph.empty
set_option maxRecDepth 4096 in
theorem node10957_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10957 :=
  Budget.node_covers _ _ 10957 row10957 node10949 Graph.empty
    node10949_covers empty_covers (by decide) (by decide) row10957_covers
theorem node10957_eq_data : node10957 = Data.node10957 := by
  have branches : (node10949, Graph.empty) = (Data.node10949, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10949_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10957 row children.1 children.2) row10957_eq_data branches
set_option maxRecDepth 4096 in
theorem row10979_eq_data : row10979 = Data.row10979 := rfl
def node10979 : Graph := .node 10979 row10979 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node10979_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10979 :=
  Budget.node_covers _ _ 10979 row10979 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row10979_covers
theorem node10979_eq_data : node10979 = Data.node10979 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10979 row children.1 children.2) row10979_eq_data branches
set_option maxRecDepth 4096 in
theorem row10987_eq_data : row10987 = Data.row10987 := rfl
def node10987 : Graph := .node 10987 row10987 node10979 Graph.empty
set_option maxRecDepth 4096 in
theorem node10987_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10987 :=
  Budget.node_covers _ _ 10987 row10987 node10979 Graph.empty
    node10979_covers empty_covers (by decide) (by decide) row10987_covers
theorem node10987_eq_data : node10987 = Data.node10987 := by
  have branches : (node10979, Graph.empty) = (Data.node10979, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10979_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10987 row children.1 children.2) row10987_eq_data branches
set_option maxRecDepth 4096 in
theorem row10973_eq_data : row10973 = Data.row10973 := rfl
def node10973 : Graph := .node 10973 row10973 node10957 node10987
set_option maxRecDepth 4096 in
theorem node10973_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10973 :=
  Budget.node_covers _ _ 10973 row10973 node10957 node10987
    node10957_covers node10987_covers (by decide) (by decide) row10973_covers
theorem node10973_eq_data : node10973 = Data.node10973 := by
  have branches : (node10957, node10987) = (Data.node10957, Data.node10987) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10957_eq_data node10987_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10973 row children.1 children.2) row10973_eq_data branches
set_option maxRecDepth 4096 in
theorem row10939_eq_data : row10939 = Data.row10939 := rfl
def node10939 : Graph := .node 10939 row10939 node10903 node10973
set_option maxRecDepth 4096 in
theorem node10939_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10939 :=
  Budget.node_covers _ _ 10939 row10939 node10903 node10973
    node10903_covers node10973_covers (by decide) (by decide) row10939_covers
theorem node10939_eq_data : node10939 = Data.node10939 := by
  have branches : (node10903, node10973) = (Data.node10903, Data.node10973) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10903_eq_data node10973_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10939 row children.1 children.2) row10939_eq_data branches
set_option maxRecDepth 4096 in
theorem row10883_eq_data : row10883 = Data.row10883 := rfl
def node10883 : Graph := .node 10883 row10883 node10837 node10939
set_option maxRecDepth 4096 in
theorem node10883_covers : Budget.Covers PrimePairVertices.largeTree 26034 node10883 :=
  Budget.node_covers _ _ 10883 row10883 node10837 node10939
    node10837_covers node10939_covers (by decide) (by decide) row10883_covers
theorem node10883_eq_data : node10883 = Data.node10883 := by
  have branches : (node10837, node10939) = (Data.node10837, Data.node10939) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node10837_eq_data node10939_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 10883 row children.1 children.2) row10883_eq_data branches
set_option maxRecDepth 4096 in
theorem row11003_eq_data : row11003 = Data.row11003 := rfl
def node11003 : Graph := .node 11003 row11003 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11003_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11003 :=
  Budget.node_covers _ _ 11003 row11003 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11003_covers
theorem node11003_eq_data : node11003 = Data.node11003 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11003 row children.1 children.2) row11003_eq_data branches
set_option maxRecDepth 4096 in
theorem row11027_eq_data : row11027 = Data.row11027 := rfl
def node11027 : Graph := .node 11027 row11027 node11003 Graph.empty
set_option maxRecDepth 4096 in
theorem node11027_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11027 :=
  Budget.node_covers _ _ 11027 row11027 node11003 Graph.empty
    node11003_covers empty_covers (by decide) (by decide) row11027_covers
theorem node11027_eq_data : node11027 = Data.node11027 := by
  have branches : (node11003, Graph.empty) = (Data.node11003, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11003_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11027 row children.1 children.2) row11027_eq_data branches
set_option maxRecDepth 4096 in
theorem row11057_eq_data : row11057 = Data.row11057 := rfl
def node11057 : Graph := .node 11057 row11057 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11057_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11057 :=
  Budget.node_covers _ _ 11057 row11057 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11057_covers
theorem node11057_eq_data : node11057 = Data.node11057 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11057 row children.1 children.2) row11057_eq_data branches
set_option maxRecDepth 4096 in
theorem row11059_eq_data : row11059 = Data.row11059 := rfl
def node11059 : Graph := .node 11059 row11059 node11057 Graph.empty
set_option maxRecDepth 4096 in
theorem node11059_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11059 :=
  Budget.node_covers _ _ 11059 row11059 node11057 Graph.empty
    node11057_covers empty_covers (by decide) (by decide) row11059_covers
theorem node11059_eq_data : node11059 = Data.node11059 := by
  have branches : (node11057, Graph.empty) = (Data.node11057, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11057_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11059 row children.1 children.2) row11059_eq_data branches
set_option maxRecDepth 4096 in
theorem row11047_eq_data : row11047 = Data.row11047 := rfl
def node11047 : Graph := .node 11047 row11047 node11027 node11059
set_option maxRecDepth 4096 in
theorem node11047_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11047 :=
  Budget.node_covers _ _ 11047 row11047 node11027 node11059
    node11027_covers node11059_covers (by decide) (by decide) row11047_covers
theorem node11047_eq_data : node11047 = Data.node11047 := by
  have branches : (node11027, node11059) = (Data.node11027, Data.node11059) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11027_eq_data node11059_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11047 row children.1 children.2) row11047_eq_data branches
set_option maxRecDepth 4096 in
theorem row11071_eq_data : row11071 = Data.row11071 := rfl
def node11071 : Graph := .node 11071 row11071 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11071_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11071 :=
  Budget.node_covers _ _ 11071 row11071 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11071_covers
theorem node11071_eq_data : node11071 = Data.node11071 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11071 row children.1 children.2) row11071_eq_data branches
set_option maxRecDepth 4096 in
theorem row11083_eq_data : row11083 = Data.row11083 := rfl
def node11083 : Graph := .node 11083 row11083 node11071 Graph.empty
set_option maxRecDepth 4096 in
theorem node11083_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11083 :=
  Budget.node_covers _ _ 11083 row11083 node11071 Graph.empty
    node11071_covers empty_covers (by decide) (by decide) row11083_covers
theorem node11083_eq_data : node11083 = Data.node11083 := by
  have branches : (node11071, Graph.empty) = (Data.node11071, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11071_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11083 row children.1 children.2) row11083_eq_data branches
set_option maxRecDepth 4096 in
theorem row11093_eq_data : row11093 = Data.row11093 := rfl
def node11093 : Graph := .node 11093 row11093 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11093_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11093 :=
  Budget.node_covers _ _ 11093 row11093 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11093_covers
theorem node11093_eq_data : node11093 = Data.node11093 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11093 row children.1 children.2) row11093_eq_data branches
set_option maxRecDepth 4096 in
theorem row11113_eq_data : row11113 = Data.row11113 := rfl
def node11113 : Graph := .node 11113 row11113 node11093 Graph.empty
set_option maxRecDepth 4096 in
theorem node11113_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11113 :=
  Budget.node_covers _ _ 11113 row11113 node11093 Graph.empty
    node11093_covers empty_covers (by decide) (by decide) row11113_covers
theorem node11113_eq_data : node11113 = Data.node11113 := by
  have branches : (node11093, Graph.empty) = (Data.node11093, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11093_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11113 row children.1 children.2) row11113_eq_data branches
set_option maxRecDepth 4096 in
theorem row11087_eq_data : row11087 = Data.row11087 := rfl
def node11087 : Graph := .node 11087 row11087 node11083 node11113
set_option maxRecDepth 4096 in
theorem node11087_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11087 :=
  Budget.node_covers _ _ 11087 row11087 node11083 node11113
    node11083_covers node11113_covers (by decide) (by decide) row11087_covers
theorem node11087_eq_data : node11087 = Data.node11087 := by
  have branches : (node11083, node11113) = (Data.node11083, Data.node11113) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11083_eq_data node11113_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11087 row children.1 children.2) row11087_eq_data branches
set_option maxRecDepth 4096 in
theorem row11069_eq_data : row11069 = Data.row11069 := rfl
def node11069 : Graph := .node 11069 row11069 node11047 node11087
set_option maxRecDepth 4096 in
theorem node11069_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11069 :=
  Budget.node_covers _ _ 11069 row11069 node11047 node11087
    node11047_covers node11087_covers (by decide) (by decide) row11069_covers
theorem node11069_eq_data : node11069 = Data.node11069 := by
  have branches : (node11047, node11087) = (Data.node11047, Data.node11087) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11047_eq_data node11087_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11069 row children.1 children.2) row11069_eq_data branches
set_option maxRecDepth 4096 in
theorem row11119_eq_data : row11119 = Data.row11119 := rfl
def node11119 : Graph := .node 11119 row11119 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11119_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11119 :=
  Budget.node_covers _ _ 11119 row11119 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11119_covers
theorem node11119_eq_data : node11119 = Data.node11119 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11119 row children.1 children.2) row11119_eq_data branches
set_option maxRecDepth 4096 in
theorem row11131_eq_data : row11131 = Data.row11131 := rfl
def node11131 : Graph := .node 11131 row11131 node11119 Graph.empty
set_option maxRecDepth 4096 in
theorem node11131_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11131 :=
  Budget.node_covers _ _ 11131 row11131 node11119 Graph.empty
    node11119_covers empty_covers (by decide) (by decide) row11131_covers
theorem node11131_eq_data : node11131 = Data.node11131 := by
  have branches : (node11119, Graph.empty) = (Data.node11119, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11119_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11131 row children.1 children.2) row11131_eq_data branches
set_option maxRecDepth 4096 in
theorem row11159_eq_data : row11159 = Data.row11159 := rfl
def node11159 : Graph := .node 11159 row11159 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node11159_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11159 :=
  Budget.node_covers _ _ 11159 row11159 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row11159_covers
theorem node11159_eq_data : node11159 = Data.node11159 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11159 row children.1 children.2) row11159_eq_data branches
set_option maxRecDepth 4096 in
theorem row11161_eq_data : row11161 = Data.row11161 := rfl
def node11161 : Graph := .node 11161 row11161 node11159 Graph.empty
set_option maxRecDepth 4096 in
theorem node11161_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11161 :=
  Budget.node_covers _ _ 11161 row11161 node11159 Graph.empty
    node11159_covers empty_covers (by decide) (by decide) row11161_covers
theorem node11161_eq_data : node11161 = Data.node11161 := by
  have branches : (node11159, Graph.empty) = (Data.node11159, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11159_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11161 row children.1 children.2) row11161_eq_data branches
set_option maxRecDepth 4096 in
theorem row11149_eq_data : row11149 = Data.row11149 := rfl
def node11149 : Graph := .node 11149 row11149 node11131 node11161
set_option maxRecDepth 4096 in
theorem node11149_covers : Budget.Covers PrimePairVertices.largeTree 26034 node11149 :=
  Budget.node_covers _ _ 11149 row11149 node11131 node11161
    node11131_covers node11161_covers (by decide) (by decide) row11149_covers
theorem node11149_eq_data : node11149 = Data.node11149 := by
  have branches : (node11131, node11161) = (Data.node11131, Data.node11161) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node11131_eq_data node11161_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 11149 row children.1 children.2) row11149_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
