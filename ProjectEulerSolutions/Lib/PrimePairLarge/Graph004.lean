import ProjectEulerSolutions.Lib.PrimePairLarge.Graph003
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
set_option maxRecDepth 4096 in
theorem row1657_eq_data : row1657 = Data.row1657 := rfl
def node1657 : Graph := .node 1657 row1657 node1637 node1667
set_option maxRecDepth 4096 in
theorem node1657_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1657 :=
  Budget.node_covers _ _ 1657 row1657 node1637 node1667
    node1637_covers node1667_covers (by decide) (by decide) row1657_covers
theorem node1657_eq_data : node1657 = Data.node1657 := by
  have branches : (node1637, node1667) = (Data.node1637, Data.node1667) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1637_eq_data node1667_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1657 row children.1 children.2) row1657_eq_data branches
set_option maxRecDepth 4096 in
theorem row1693_eq_data : row1693 = Data.row1693 := rfl
def node1693 : Graph := .node 1693 row1693 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1693_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1693 :=
  Budget.node_covers _ _ 1693 row1693 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1693_covers
theorem node1693_eq_data : node1693 = Data.node1693 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1693 row children.1 children.2) row1693_eq_data branches
set_option maxRecDepth 4096 in
theorem row1697_eq_data : row1697 = Data.row1697 := rfl
def node1697 : Graph := .node 1697 row1697 node1693 Graph.empty
set_option maxRecDepth 4096 in
theorem node1697_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1697 :=
  Budget.node_covers _ _ 1697 row1697 node1693 Graph.empty
    node1693_covers empty_covers (by decide) (by decide) row1697_covers
theorem node1697_eq_data : node1697 = Data.node1697 := by
  have branches : (node1693, Graph.empty) = (Data.node1693, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1693_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1697 row children.1 children.2) row1697_eq_data branches
set_option maxRecDepth 4096 in
theorem row1709_eq_data : row1709 = Data.row1709 := rfl
def node1709 : Graph := .node 1709 row1709 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1709_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1709 :=
  Budget.node_covers _ _ 1709 row1709 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1709_covers
theorem node1709_eq_data : node1709 = Data.node1709 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1709 row children.1 children.2) row1709_eq_data branches
set_option maxRecDepth 4096 in
theorem row1721_eq_data : row1721 = Data.row1721 := rfl
def node1721 : Graph := .node 1721 row1721 node1709 Graph.empty
set_option maxRecDepth 4096 in
theorem node1721_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1721 :=
  Budget.node_covers _ _ 1721 row1721 node1709 Graph.empty
    node1709_covers empty_covers (by decide) (by decide) row1721_covers
theorem node1721_eq_data : node1721 = Data.node1721 := by
  have branches : (node1709, Graph.empty) = (Data.node1709, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1709_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1721 row children.1 children.2) row1721_eq_data branches
set_option maxRecDepth 4096 in
theorem row1699_eq_data : row1699 = Data.row1699 := rfl
def node1699 : Graph := .node 1699 row1699 node1697 node1721
set_option maxRecDepth 4096 in
theorem node1699_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1699 :=
  Budget.node_covers _ _ 1699 row1699 node1697 node1721
    node1697_covers node1721_covers (by decide) (by decide) row1699_covers
theorem node1699_eq_data : node1699 = Data.node1699 := by
  have branches : (node1697, node1721) = (Data.node1697, Data.node1721) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1697_eq_data node1721_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1699 row children.1 children.2) row1699_eq_data branches
set_option maxRecDepth 4096 in
theorem row1669_eq_data : row1669 = Data.row1669 := rfl
def node1669 : Graph := .node 1669 row1669 node1657 node1699
set_option maxRecDepth 4096 in
theorem node1669_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1669 :=
  Budget.node_covers _ _ 1669 row1669 node1657 node1699
    node1657_covers node1699_covers (by decide) (by decide) row1669_covers
theorem node1669_eq_data : node1669 = Data.node1669 := by
  have branches : (node1657, node1699) = (Data.node1657, Data.node1699) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1657_eq_data node1699_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1669 row children.1 children.2) row1669_eq_data branches
set_option maxRecDepth 4096 in
theorem row1621_eq_data : row1621 = Data.row1621 := rfl
def node1621 : Graph := .node 1621 row1621 node1597 node1669
set_option maxRecDepth 4096 in
theorem node1621_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1621 :=
  Budget.node_covers _ _ 1621 row1621 node1597 node1669
    node1597_covers node1669_covers (by decide) (by decide) row1621_covers
theorem node1621_eq_data : node1621 = Data.node1621 := by
  have branches : (node1597, node1669) = (Data.node1597, Data.node1669) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1597_eq_data node1669_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1621 row children.1 children.2) row1621_eq_data branches
set_option maxRecDepth 4096 in
theorem row1733_eq_data : row1733 = Data.row1733 := rfl
def node1733 : Graph := .node 1733 row1733 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1733_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1733 :=
  Budget.node_covers _ _ 1733 row1733 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1733_covers
theorem node1733_eq_data : node1733 = Data.node1733 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1733 row children.1 children.2) row1733_eq_data branches
set_option maxRecDepth 4096 in
theorem row1741_eq_data : row1741 = Data.row1741 := rfl
def node1741 : Graph := .node 1741 row1741 node1733 Graph.empty
set_option maxRecDepth 4096 in
theorem node1741_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1741 :=
  Budget.node_covers _ _ 1741 row1741 node1733 Graph.empty
    node1733_covers empty_covers (by decide) (by decide) row1741_covers
theorem node1741_eq_data : node1741 = Data.node1741 := by
  have branches : (node1733, Graph.empty) = (Data.node1733, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1733_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1741 row children.1 children.2) row1741_eq_data branches
set_option maxRecDepth 4096 in
theorem row1753_eq_data : row1753 = Data.row1753 := rfl
def node1753 : Graph := .node 1753 row1753 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1753_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1753 :=
  Budget.node_covers _ _ 1753 row1753 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1753_covers
theorem node1753_eq_data : node1753 = Data.node1753 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1753 row children.1 children.2) row1753_eq_data branches
set_option maxRecDepth 4096 in
theorem row1759_eq_data : row1759 = Data.row1759 := rfl
def node1759 : Graph := .node 1759 row1759 node1753 Graph.empty
set_option maxRecDepth 4096 in
theorem node1759_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1759 :=
  Budget.node_covers _ _ 1759 row1759 node1753 Graph.empty
    node1753_covers empty_covers (by decide) (by decide) row1759_covers
theorem node1759_eq_data : node1759 = Data.node1759 := by
  have branches : (node1753, Graph.empty) = (Data.node1753, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1753_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1759 row children.1 children.2) row1759_eq_data branches
set_option maxRecDepth 4096 in
theorem row1747_eq_data : row1747 = Data.row1747 := rfl
def node1747 : Graph := .node 1747 row1747 node1741 node1759
set_option maxRecDepth 4096 in
theorem node1747_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1747 :=
  Budget.node_covers _ _ 1747 row1747 node1741 node1759
    node1741_covers node1759_covers (by decide) (by decide) row1747_covers
theorem node1747_eq_data : node1747 = Data.node1747 := by
  have branches : (node1741, node1759) = (Data.node1741, Data.node1759) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1741_eq_data node1759_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1747 row children.1 children.2) row1747_eq_data branches
set_option maxRecDepth 4096 in
theorem row1783_eq_data : row1783 = Data.row1783 := rfl
def node1783 : Graph := .node 1783 row1783 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1783_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1783 :=
  Budget.node_covers _ _ 1783 row1783 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1783_covers
theorem node1783_eq_data : node1783 = Data.node1783 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1783 row children.1 children.2) row1783_eq_data branches
set_option maxRecDepth 4096 in
theorem row1787_eq_data : row1787 = Data.row1787 := rfl
def node1787 : Graph := .node 1787 row1787 node1783 Graph.empty
set_option maxRecDepth 4096 in
theorem node1787_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1787 :=
  Budget.node_covers _ _ 1787 row1787 node1783 Graph.empty
    node1783_covers empty_covers (by decide) (by decide) row1787_covers
theorem node1787_eq_data : node1787 = Data.node1787 := by
  have branches : (node1783, Graph.empty) = (Data.node1783, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1783_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1787 row children.1 children.2) row1787_eq_data branches
set_option maxRecDepth 4096 in
theorem row1801_eq_data : row1801 = Data.row1801 := rfl
def node1801 : Graph := .node 1801 row1801 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1801_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1801 :=
  Budget.node_covers _ _ 1801 row1801 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1801_covers
theorem node1801_eq_data : node1801 = Data.node1801 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1801 row children.1 children.2) row1801_eq_data branches
set_option maxRecDepth 4096 in
theorem row1811_eq_data : row1811 = Data.row1811 := rfl
def node1811 : Graph := .node 1811 row1811 node1801 Graph.empty
set_option maxRecDepth 4096 in
theorem node1811_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1811 :=
  Budget.node_covers _ _ 1811 row1811 node1801 Graph.empty
    node1801_covers empty_covers (by decide) (by decide) row1811_covers
theorem node1811_eq_data : node1811 = Data.node1811 := by
  have branches : (node1801, Graph.empty) = (Data.node1801, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1801_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1811 row children.1 children.2) row1811_eq_data branches
set_option maxRecDepth 4096 in
theorem row1789_eq_data : row1789 = Data.row1789 := rfl
def node1789 : Graph := .node 1789 row1789 node1787 node1811
set_option maxRecDepth 4096 in
theorem node1789_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1789 :=
  Budget.node_covers _ _ 1789 row1789 node1787 node1811
    node1787_covers node1811_covers (by decide) (by decide) row1789_covers
theorem node1789_eq_data : node1789 = Data.node1789 := by
  have branches : (node1787, node1811) = (Data.node1787, Data.node1811) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1787_eq_data node1811_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1789 row children.1 children.2) row1789_eq_data branches
set_option maxRecDepth 4096 in
theorem row1777_eq_data : row1777 = Data.row1777 := rfl
def node1777 : Graph := .node 1777 row1777 node1747 node1789
set_option maxRecDepth 4096 in
theorem node1777_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1777 :=
  Budget.node_covers _ _ 1777 row1777 node1747 node1789
    node1747_covers node1789_covers (by decide) (by decide) row1777_covers
theorem node1777_eq_data : node1777 = Data.node1777 := by
  have branches : (node1747, node1789) = (Data.node1747, Data.node1789) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1747_eq_data node1789_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1777 row children.1 children.2) row1777_eq_data branches
set_option maxRecDepth 4096 in
theorem row1831_eq_data : row1831 = Data.row1831 := rfl
def node1831 : Graph := .node 1831 row1831 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1831_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1831 :=
  Budget.node_covers _ _ 1831 row1831 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1831_covers
theorem node1831_eq_data : node1831 = Data.node1831 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1831 row children.1 children.2) row1831_eq_data branches
set_option maxRecDepth 4096 in
theorem row1847_eq_data : row1847 = Data.row1847 := rfl
def node1847 : Graph := .node 1847 row1847 node1831 Graph.empty
set_option maxRecDepth 4096 in
theorem node1847_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1847 :=
  Budget.node_covers _ _ 1847 row1847 node1831 Graph.empty
    node1831_covers empty_covers (by decide) (by decide) row1847_covers
theorem node1847_eq_data : node1847 = Data.node1847 := by
  have branches : (node1831, Graph.empty) = (Data.node1831, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1831_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1847 row children.1 children.2) row1847_eq_data branches
set_option maxRecDepth 4096 in
theorem row1867_eq_data : row1867 = Data.row1867 := rfl
def node1867 : Graph := .node 1867 row1867 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1867_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1867 :=
  Budget.node_covers _ _ 1867 row1867 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1867_covers
theorem node1867_eq_data : node1867 = Data.node1867 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1867 row children.1 children.2) row1867_eq_data branches
set_option maxRecDepth 4096 in
theorem row1871_eq_data : row1871 = Data.row1871 := rfl
def node1871 : Graph := .node 1871 row1871 node1867 Graph.empty
set_option maxRecDepth 4096 in
theorem node1871_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1871 :=
  Budget.node_covers _ _ 1871 row1871 node1867 Graph.empty
    node1867_covers empty_covers (by decide) (by decide) row1871_covers
theorem node1871_eq_data : node1871 = Data.node1871 := by
  have branches : (node1867, Graph.empty) = (Data.node1867, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1867_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1871 row children.1 children.2) row1871_eq_data branches
set_option maxRecDepth 4096 in
theorem row1861_eq_data : row1861 = Data.row1861 := rfl
def node1861 : Graph := .node 1861 row1861 node1847 node1871
set_option maxRecDepth 4096 in
theorem node1861_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1861 :=
  Budget.node_covers _ _ 1861 row1861 node1847 node1871
    node1847_covers node1871_covers (by decide) (by decide) row1861_covers
theorem node1861_eq_data : node1861 = Data.node1861 := by
  have branches : (node1847, node1871) = (Data.node1847, Data.node1871) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1847_eq_data node1871_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1861 row children.1 children.2) row1861_eq_data branches
set_option maxRecDepth 4096 in
theorem row1877_eq_data : row1877 = Data.row1877 := rfl
def node1877 : Graph := .node 1877 row1877 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1877_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1877 :=
  Budget.node_covers _ _ 1877 row1877 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1877_covers
theorem node1877_eq_data : node1877 = Data.node1877 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1877 row children.1 children.2) row1877_eq_data branches
set_option maxRecDepth 4096 in
theorem row1879_eq_data : row1879 = Data.row1879 := rfl
def node1879 : Graph := .node 1879 row1879 node1877 Graph.empty
set_option maxRecDepth 4096 in
theorem node1879_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1879 :=
  Budget.node_covers _ _ 1879 row1879 node1877 Graph.empty
    node1877_covers empty_covers (by decide) (by decide) row1879_covers
theorem node1879_eq_data : node1879 = Data.node1879 := by
  have branches : (node1877, Graph.empty) = (Data.node1877, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1877_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1879 row children.1 children.2) row1879_eq_data branches
set_option maxRecDepth 4096 in
theorem row1901_eq_data : row1901 = Data.row1901 := rfl
def node1901 : Graph := .node 1901 row1901 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1901_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1901 :=
  Budget.node_covers _ _ 1901 row1901 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1901_covers
theorem node1901_eq_data : node1901 = Data.node1901 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1901 row children.1 children.2) row1901_eq_data branches
set_option maxRecDepth 4096 in
theorem row1907_eq_data : row1907 = Data.row1907 := rfl
def node1907 : Graph := .node 1907 row1907 node1901 Graph.empty
set_option maxRecDepth 4096 in
theorem node1907_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1907 :=
  Budget.node_covers _ _ 1907 row1907 node1901 Graph.empty
    node1901_covers empty_covers (by decide) (by decide) row1907_covers
theorem node1907_eq_data : node1907 = Data.node1907 := by
  have branches : (node1901, Graph.empty) = (Data.node1901, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1901_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1907 row children.1 children.2) row1907_eq_data branches
set_option maxRecDepth 4096 in
theorem row1889_eq_data : row1889 = Data.row1889 := rfl
def node1889 : Graph := .node 1889 row1889 node1879 node1907
set_option maxRecDepth 4096 in
theorem node1889_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1889 :=
  Budget.node_covers _ _ 1889 row1889 node1879 node1907
    node1879_covers node1907_covers (by decide) (by decide) row1889_covers
theorem node1889_eq_data : node1889 = Data.node1889 := by
  have branches : (node1879, node1907) = (Data.node1879, Data.node1907) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1879_eq_data node1907_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1889 row children.1 children.2) row1889_eq_data branches
set_option maxRecDepth 4096 in
theorem row1873_eq_data : row1873 = Data.row1873 := rfl
def node1873 : Graph := .node 1873 row1873 node1861 node1889
set_option maxRecDepth 4096 in
theorem node1873_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1873 :=
  Budget.node_covers _ _ 1873 row1873 node1861 node1889
    node1861_covers node1889_covers (by decide) (by decide) row1873_covers
theorem node1873_eq_data : node1873 = Data.node1873 := by
  have branches : (node1861, node1889) = (Data.node1861, Data.node1889) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1861_eq_data node1889_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1873 row children.1 children.2) row1873_eq_data branches
set_option maxRecDepth 4096 in
theorem row1823_eq_data : row1823 = Data.row1823 := rfl
def node1823 : Graph := .node 1823 row1823 node1777 node1873
set_option maxRecDepth 4096 in
theorem node1823_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1823 :=
  Budget.node_covers _ _ 1823 row1823 node1777 node1873
    node1777_covers node1873_covers (by decide) (by decide) row1823_covers
theorem node1823_eq_data : node1823 = Data.node1823 := by
  have branches : (node1777, node1873) = (Data.node1777, Data.node1873) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1777_eq_data node1873_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1823 row children.1 children.2) row1823_eq_data branches
set_option maxRecDepth 4096 in
theorem row1723_eq_data : row1723 = Data.row1723 := rfl
def node1723 : Graph := .node 1723 row1723 node1621 node1823
set_option maxRecDepth 4096 in
theorem node1723_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1723 :=
  Budget.node_covers _ _ 1723 row1723 node1621 node1823
    node1621_covers node1823_covers (by decide) (by decide) row1723_covers
theorem node1723_eq_data : node1723 = Data.node1723 := by
  have branches : (node1621, node1823) = (Data.node1621, Data.node1823) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1621_eq_data node1823_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1723 row children.1 children.2) row1723_eq_data branches
set_option maxRecDepth 4096 in
theorem row1553_eq_data : row1553 = Data.row1553 := rfl
def node1553 : Graph := .node 1553 row1553 node1381 node1723
set_option maxRecDepth 4096 in
theorem node1553_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1553 :=
  Budget.node_covers _ _ 1553 row1553 node1381 node1723
    node1381_covers node1723_covers (by decide) (by decide) row1553_covers
theorem node1553_eq_data : node1553 = Data.node1553 := by
  have branches : (node1381, node1723) = (Data.node1381, Data.node1723) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1381_eq_data node1723_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1553 row children.1 children.2) row1553_eq_data branches
set_option maxRecDepth 4096 in
theorem row1931_eq_data : row1931 = Data.row1931 := rfl
def node1931 : Graph := .node 1931 row1931 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1931_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1931 :=
  Budget.node_covers _ _ 1931 row1931 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1931_covers
theorem node1931_eq_data : node1931 = Data.node1931 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1931 row children.1 children.2) row1931_eq_data branches
set_option maxRecDepth 4096 in
theorem row1949_eq_data : row1949 = Data.row1949 := rfl
def node1949 : Graph := .node 1949 row1949 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1949_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1949 :=
  Budget.node_covers _ _ 1949 row1949 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1949_covers
theorem node1949_eq_data : node1949 = Data.node1949 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1949 row children.1 children.2) row1949_eq_data branches
set_option maxRecDepth 4096 in
theorem row1933_eq_data : row1933 = Data.row1933 := rfl
def node1933 : Graph := .node 1933 row1933 node1931 node1949
set_option maxRecDepth 4096 in
theorem node1933_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1933 :=
  Budget.node_covers _ _ 1933 row1933 node1931 node1949
    node1931_covers node1949_covers (by decide) (by decide) row1933_covers
theorem node1933_eq_data : node1933 = Data.node1933 := by
  have branches : (node1931, node1949) = (Data.node1931, Data.node1949) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1931_eq_data node1949_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1933 row children.1 children.2) row1933_eq_data branches
set_option maxRecDepth 4096 in
theorem row1973_eq_data : row1973 = Data.row1973 := rfl
def node1973 : Graph := .node 1973 row1973 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1973_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1973 :=
  Budget.node_covers _ _ 1973 row1973 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1973_covers
theorem node1973_eq_data : node1973 = Data.node1973 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1973 row children.1 children.2) row1973_eq_data branches
set_option maxRecDepth 4096 in
theorem row1979_eq_data : row1979 = Data.row1979 := rfl
def node1979 : Graph := .node 1979 row1979 node1973 Graph.empty
set_option maxRecDepth 4096 in
theorem node1979_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1979 :=
  Budget.node_covers _ _ 1979 row1979 node1973 Graph.empty
    node1973_covers empty_covers (by decide) (by decide) row1979_covers
theorem node1979_eq_data : node1979 = Data.node1979 := by
  have branches : (node1973, Graph.empty) = (Data.node1973, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1973_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1979 row children.1 children.2) row1979_eq_data branches
set_option maxRecDepth 4096 in
theorem row1951_eq_data : row1951 = Data.row1951 := rfl
def node1951 : Graph := .node 1951 row1951 node1933 node1979
set_option maxRecDepth 4096 in
theorem node1951_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1951 :=
  Budget.node_covers _ _ 1951 row1951 node1933 node1979
    node1933_covers node1979_covers (by decide) (by decide) row1951_covers
theorem node1951_eq_data : node1951 = Data.node1951 := by
  have branches : (node1933, node1979) = (Data.node1933, Data.node1979) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1933_eq_data node1979_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1951 row children.1 children.2) row1951_eq_data branches
set_option maxRecDepth 4096 in
theorem row1993_eq_data : row1993 = Data.row1993 := rfl
def node1993 : Graph := .node 1993 row1993 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node1993_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1993 :=
  Budget.node_covers _ _ 1993 row1993 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row1993_covers
theorem node1993_eq_data : node1993 = Data.node1993 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1993 row children.1 children.2) row1993_eq_data branches
set_option maxRecDepth 4096 in
theorem row1997_eq_data : row1997 = Data.row1997 := rfl
def node1997 : Graph := .node 1997 row1997 node1993 Graph.empty
set_option maxRecDepth 4096 in
theorem node1997_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1997 :=
  Budget.node_covers _ _ 1997 row1997 node1993 Graph.empty
    node1993_covers empty_covers (by decide) (by decide) row1997_covers
theorem node1997_eq_data : node1997 = Data.node1997 := by
  have branches : (node1993, Graph.empty) = (Data.node1993, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1993_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1997 row children.1 children.2) row1997_eq_data branches
set_option maxRecDepth 4096 in
theorem row2003_eq_data : row2003 = Data.row2003 := rfl
def node2003 : Graph := .node 2003 row2003 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2003_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2003 :=
  Budget.node_covers _ _ 2003 row2003 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2003_covers
theorem node2003_eq_data : node2003 = Data.node2003 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2003 row children.1 children.2) row2003_eq_data branches
set_option maxRecDepth 4096 in
theorem row2011_eq_data : row2011 = Data.row2011 := rfl
def node2011 : Graph := .node 2011 row2011 node2003 Graph.empty
set_option maxRecDepth 4096 in
theorem node2011_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2011 :=
  Budget.node_covers _ _ 2011 row2011 node2003 Graph.empty
    node2003_covers empty_covers (by decide) (by decide) row2011_covers
theorem node2011_eq_data : node2011 = Data.node2011 := by
  have branches : (node2003, Graph.empty) = (Data.node2003, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2003_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2011 row children.1 children.2) row2011_eq_data branches
set_option maxRecDepth 4096 in
theorem row1999_eq_data : row1999 = Data.row1999 := rfl
def node1999 : Graph := .node 1999 row1999 node1997 node2011
set_option maxRecDepth 4096 in
theorem node1999_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1999 :=
  Budget.node_covers _ _ 1999 row1999 node1997 node2011
    node1997_covers node2011_covers (by decide) (by decide) row1999_covers
theorem node1999_eq_data : node1999 = Data.node1999 := by
  have branches : (node1997, node2011) = (Data.node1997, Data.node2011) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1997_eq_data node2011_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1999 row children.1 children.2) row1999_eq_data branches
set_option maxRecDepth 4096 in
theorem row1987_eq_data : row1987 = Data.row1987 := rfl
def node1987 : Graph := .node 1987 row1987 node1951 node1999
set_option maxRecDepth 4096 in
theorem node1987_covers : Budget.Covers PrimePairVertices.largeTree 26034 node1987 :=
  Budget.node_covers _ _ 1987 row1987 node1951 node1999
    node1951_covers node1999_covers (by decide) (by decide) row1987_covers
theorem node1987_eq_data : node1987 = Data.node1987 := by
  have branches : (node1951, node1999) = (Data.node1951, Data.node1999) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1951_eq_data node1999_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 1987 row children.1 children.2) row1987_eq_data branches
set_option maxRecDepth 4096 in
theorem row2027_eq_data : row2027 = Data.row2027 := rfl
def node2027 : Graph := .node 2027 row2027 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2027_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2027 :=
  Budget.node_covers _ _ 2027 row2027 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2027_covers
theorem node2027_eq_data : node2027 = Data.node2027 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2027 row children.1 children.2) row2027_eq_data branches
set_option maxRecDepth 4096 in
theorem row2029_eq_data : row2029 = Data.row2029 := rfl
def node2029 : Graph := .node 2029 row2029 node2027 Graph.empty
set_option maxRecDepth 4096 in
theorem node2029_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2029 :=
  Budget.node_covers _ _ 2029 row2029 node2027 Graph.empty
    node2027_covers empty_covers (by decide) (by decide) row2029_covers
theorem node2029_eq_data : node2029 = Data.node2029 := by
  have branches : (node2027, Graph.empty) = (Data.node2027, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2027_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2029 row children.1 children.2) row2029_eq_data branches
set_option maxRecDepth 4096 in
theorem row2053_eq_data : row2053 = Data.row2053 := rfl
def node2053 : Graph := .node 2053 row2053 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2053_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2053 :=
  Budget.node_covers _ _ 2053 row2053 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2053_covers
theorem node2053_eq_data : node2053 = Data.node2053 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2053 row children.1 children.2) row2053_eq_data branches
set_option maxRecDepth 4096 in
theorem row2063_eq_data : row2063 = Data.row2063 := rfl
def node2063 : Graph := .node 2063 row2063 node2053 Graph.empty
set_option maxRecDepth 4096 in
theorem node2063_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2063 :=
  Budget.node_covers _ _ 2063 row2063 node2053 Graph.empty
    node2053_covers empty_covers (by decide) (by decide) row2063_covers
theorem node2063_eq_data : node2063 = Data.node2063 := by
  have branches : (node2053, Graph.empty) = (Data.node2053, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2053_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2063 row children.1 children.2) row2063_eq_data branches
set_option maxRecDepth 4096 in
theorem row2039_eq_data : row2039 = Data.row2039 := rfl
def node2039 : Graph := .node 2039 row2039 node2029 node2063
set_option maxRecDepth 4096 in
theorem node2039_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2039 :=
  Budget.node_covers _ _ 2039 row2039 node2029 node2063
    node2029_covers node2063_covers (by decide) (by decide) row2039_covers
theorem node2039_eq_data : node2039 = Data.node2039 := by
  have branches : (node2029, node2063) = (Data.node2029, Data.node2063) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2029_eq_data node2063_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2039 row children.1 children.2) row2039_eq_data branches
set_option maxRecDepth 4096 in
theorem row2081_eq_data : row2081 = Data.row2081 := rfl
def node2081 : Graph := .node 2081 row2081 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2081_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2081 :=
  Budget.node_covers _ _ 2081 row2081 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2081_covers
theorem node2081_eq_data : node2081 = Data.node2081 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2081 row children.1 children.2) row2081_eq_data branches
set_option maxRecDepth 4096 in
theorem row2083_eq_data : row2083 = Data.row2083 := rfl
def node2083 : Graph := .node 2083 row2083 node2081 Graph.empty
set_option maxRecDepth 4096 in
theorem node2083_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2083 :=
  Budget.node_covers _ _ 2083 row2083 node2081 Graph.empty
    node2081_covers empty_covers (by decide) (by decide) row2083_covers
theorem node2083_eq_data : node2083 = Data.node2083 := by
  have branches : (node2081, Graph.empty) = (Data.node2081, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2081_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2083 row children.1 children.2) row2083_eq_data branches
set_option maxRecDepth 4096 in
theorem row2089_eq_data : row2089 = Data.row2089 := rfl
def node2089 : Graph := .node 2089 row2089 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2089_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2089 :=
  Budget.node_covers _ _ 2089 row2089 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2089_covers
theorem node2089_eq_data : node2089 = Data.node2089 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2089 row children.1 children.2) row2089_eq_data branches
set_option maxRecDepth 4096 in
theorem row2099_eq_data : row2099 = Data.row2099 := rfl
def node2099 : Graph := .node 2099 row2099 node2089 Graph.empty
set_option maxRecDepth 4096 in
theorem node2099_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2099 :=
  Budget.node_covers _ _ 2099 row2099 node2089 Graph.empty
    node2089_covers empty_covers (by decide) (by decide) row2099_covers
theorem node2099_eq_data : node2099 = Data.node2099 := by
  have branches : (node2089, Graph.empty) = (Data.node2089, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2089_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2099 row children.1 children.2) row2099_eq_data branches
set_option maxRecDepth 4096 in
theorem row2087_eq_data : row2087 = Data.row2087 := rfl
def node2087 : Graph := .node 2087 row2087 node2083 node2099
set_option maxRecDepth 4096 in
theorem node2087_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2087 :=
  Budget.node_covers _ _ 2087 row2087 node2083 node2099
    node2083_covers node2099_covers (by decide) (by decide) row2087_covers
theorem node2087_eq_data : node2087 = Data.node2087 := by
  have branches : (node2083, node2099) = (Data.node2083, Data.node2099) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2083_eq_data node2099_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2087 row children.1 children.2) row2087_eq_data branches
set_option maxRecDepth 4096 in
theorem row2069_eq_data : row2069 = Data.row2069 := rfl
def node2069 : Graph := .node 2069 row2069 node2039 node2087
set_option maxRecDepth 4096 in
theorem node2069_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2069 :=
  Budget.node_covers _ _ 2069 row2069 node2039 node2087
    node2039_covers node2087_covers (by decide) (by decide) row2069_covers
theorem node2069_eq_data : node2069 = Data.node2069 := by
  have branches : (node2039, node2087) = (Data.node2039, Data.node2087) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2039_eq_data node2087_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2069 row children.1 children.2) row2069_eq_data branches
set_option maxRecDepth 4096 in
theorem row2017_eq_data : row2017 = Data.row2017 := rfl
def node2017 : Graph := .node 2017 row2017 node1987 node2069
set_option maxRecDepth 4096 in
theorem node2017_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2017 :=
  Budget.node_covers _ _ 2017 row2017 node1987 node2069
    node1987_covers node2069_covers (by decide) (by decide) row2017_covers
theorem node2017_eq_data : node2017 = Data.node2017 := by
  have branches : (node1987, node2069) = (Data.node1987, Data.node2069) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node1987_eq_data node2069_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2017 row children.1 children.2) row2017_eq_data branches
set_option maxRecDepth 4096 in
theorem row2113_eq_data : row2113 = Data.row2113 := rfl
def node2113 : Graph := .node 2113 row2113 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2113_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2113 :=
  Budget.node_covers _ _ 2113 row2113 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2113_covers
theorem node2113_eq_data : node2113 = Data.node2113 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2113 row children.1 children.2) row2113_eq_data branches
set_option maxRecDepth 4096 in
theorem row2129_eq_data : row2129 = Data.row2129 := rfl
def node2129 : Graph := .node 2129 row2129 node2113 Graph.empty
set_option maxRecDepth 4096 in
theorem node2129_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2129 :=
  Budget.node_covers _ _ 2129 row2129 node2113 Graph.empty
    node2113_covers empty_covers (by decide) (by decide) row2129_covers
theorem node2129_eq_data : node2129 = Data.node2129 := by
  have branches : (node2113, Graph.empty) = (Data.node2113, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2113_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2129 row children.1 children.2) row2129_eq_data branches
set_option maxRecDepth 4096 in
theorem row2137_eq_data : row2137 = Data.row2137 := rfl
def node2137 : Graph := .node 2137 row2137 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2137_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2137 :=
  Budget.node_covers _ _ 2137 row2137 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2137_covers
theorem node2137_eq_data : node2137 = Data.node2137 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2137 row children.1 children.2) row2137_eq_data branches
set_option maxRecDepth 4096 in
theorem row2141_eq_data : row2141 = Data.row2141 := rfl
def node2141 : Graph := .node 2141 row2141 node2137 Graph.empty
set_option maxRecDepth 4096 in
theorem node2141_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2141 :=
  Budget.node_covers _ _ 2141 row2141 node2137 Graph.empty
    node2137_covers empty_covers (by decide) (by decide) row2141_covers
theorem node2141_eq_data : node2141 = Data.node2141 := by
  have branches : (node2137, Graph.empty) = (Data.node2137, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2137_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2141 row children.1 children.2) row2141_eq_data branches
set_option maxRecDepth 4096 in
theorem row2131_eq_data : row2131 = Data.row2131 := rfl
def node2131 : Graph := .node 2131 row2131 node2129 node2141
set_option maxRecDepth 4096 in
theorem node2131_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2131 :=
  Budget.node_covers _ _ 2131 row2131 node2129 node2141
    node2129_covers node2141_covers (by decide) (by decide) row2131_covers
theorem node2131_eq_data : node2131 = Data.node2131 := by
  have branches : (node2129, node2141) = (Data.node2129, Data.node2141) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2129_eq_data node2141_eq_data
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2131 row children.1 children.2) row2131_eq_data branches
set_option maxRecDepth 4096 in
theorem row2153_eq_data : row2153 = Data.row2153 := rfl
def node2153 : Graph := .node 2153 row2153 Graph.empty Graph.empty
set_option maxRecDepth 4096 in
theorem node2153_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2153 :=
  Budget.node_covers _ _ 2153 row2153 Graph.empty Graph.empty
    empty_covers empty_covers (by decide) (by decide) row2153_covers
theorem node2153_eq_data : node2153 = Data.node2153 := by
  have branches : (Graph.empty, Graph.empty) = (Graph.empty, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) rfl rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2153 row children.1 children.2) row2153_eq_data branches
set_option maxRecDepth 4096 in
theorem row2161_eq_data : row2161 = Data.row2161 := rfl
def node2161 : Graph := .node 2161 row2161 node2153 Graph.empty
set_option maxRecDepth 4096 in
theorem node2161_covers : Budget.Covers PrimePairVertices.largeTree 26034 node2161 :=
  Budget.node_covers _ _ 2161 row2161 node2153 Graph.empty
    node2153_covers empty_covers (by decide) (by decide) row2161_covers
theorem node2161_eq_data : node2161 = Data.node2161 := by
  have branches : (node2153, Graph.empty) = (Data.node2153, Graph.empty) :=
    congrArg₂ (fun (a b : Graph) => (a, b)) node2153_eq_data rfl
  exact congrArg₂ (fun (row : List Nat) (children : Graph × Graph) =>
    Graph.node 2161 row children.1 children.2) row2161_eq_data branches
end ProjectEulerSolutions.PrimePairLarge
