import ProjectEulerSolutions.Lib.PrimePairCertificate
import Mathlib.Data.List.Sort
import ProjectEulerSolutions.Lib.CliqueSearchAgreement

namespace ProjectEulerSolutions.PrimePairCertificate

/-- Balanced input blocks keep certificate reduction shallow. -/
inductive VertexTree where
  | leaf (values : List Nat)
  | branch (left right : VertexTree)

def VertexTree.values : VertexTree → List Nat
  | .leaf values => values
  | .branch left right => left.values ++ right.values

def VertexTree.filter (predicate : Nat → Bool) : VertexTree → List Nat
  | .leaf values => values.filter predicate
  | .branch left right => left.filter predicate ++ right.filter predicate

theorem VertexTree.filter_eq (tree : VertexTree) (predicate : Nat → Bool) :
    tree.filter predicate = tree.values.filter predicate := by
  induction tree with
  | leaf values => rfl
  | branch left right ihLeft ihRight =>
    simp only [filter, values, ihLeft, ihRight, List.filter_append]

/-- Only increasing vertex pairs are queried by the canonical search. -/
def upperNeighbors (tree : VertexTree) (p : Nat) : List Nat :=
  tree.filter (fun q => decide (p < q) && relation p q)

theorem mem_upperNeighbors (tree : VertexTree) (p q : Nat) :
    q ∈ upperNeighbors tree p ↔ q ∈ tree.values ∧ p < q ∧ relation p q = true := by
  simp only [upperNeighbors, VertexTree.filter_eq, List.mem_filter, Bool.and_eq_true, decide_eq_true_eq]

/-- The cache uses a binary search tree; numeric row checks remain separate. -/
inductive Graph where
  | empty
  | node (p : Nat) (neighbors : List Nat) (left right : Graph)

def Graph.vertices : Graph → List Nat
  | .empty => []
  | .node p _ left right => left.vertices ++ p :: right.vertices

def Graph.edge : Graph → Nat → Nat → Bool
  | .empty, _, _ => false
  | .node p neighbors left right, query, q =>
    if query < p then left.edge query q else
      if p < query then right.edge query q else neighbors.contains q

def Graph.Agrees (tree : VertexTree) (graph : Graph) : Prop :=
  ∀ p ∈ graph.vertices, ∀ q ∈ tree.values, p < q → graph.edge p q = relation p q

theorem Graph.empty_agrees (tree : VertexTree) : Graph.Agrees tree .empty := by
  intro p member
  simp [vertices] at member

/-- Combining certified rows never recomputes a primality test. Bounds on
child keys justify the binary lookup for every key represented by the cache. -/
theorem Graph.node_agrees (tree : VertexTree) (p : Nat) (neighbors : List Nat) (left right : Graph)
    (leftAgrees : left.Agrees tree) (rightAgrees : right.Agrees tree)
    (before : ∀ value ∈ left.vertices, value < p)
    (after : ∀ value ∈ right.vertices, p < value)
    (checked : upperNeighbors tree p = neighbors) :
    Graph.Agrees tree (.node p neighbors left right) := by
  intro query member q memberQ order
  simp only [vertices, List.mem_append, List.mem_cons] at member
  rcases member with inLeft | same | inRight
  · have lower := before query inLeft
    simp only [edge, lower, ↓reduceIte]
    exact leftAgrees query inLeft q memberQ order
  · subst query
    simp only [edge, Nat.lt_irrefl, ↓reduceIte]
    rw [Bool.eq_iff_iff, List.contains_iff_mem, ← checked, mem_upperNeighbors]
    simp only [memberQ, order, true_and]
  · have upper := after query inRight
    simp only [edge, show ¬ query < p by omega, upper, ↓reduceIte]
    exact rightAgrees query inRight q memberQ order

/-- A complete certified cache preserves the exact minimum and selected
list, including ties and an unchanged sentinel. -/
theorem Graph.search_eq (tree : VertexTree) (graph : Graph) (agrees : graph.Agrees tree)
    (coverage : ∀ p ∈ tree.values, p ∈ graph.vertices) (ordered : tree.values.Pairwise (· < ·))
    (need : Nat) (before : CliqueSearch.Best) :
    CliqueSearch.search relation need [] tree.values 0 before =
      CliqueSearch.search graph.edge need [] tree.values 0 before := by
  have agreement : CliqueSearch.AgreeBelow relation graph.edge tree.values tree.values.sum := by
    intro p memberP q memberQ order _
    exact (agrees p (coverage p memberP) q memberQ order).symm
  have bounded : ∀ p ∈ tree.values, p ≤ tree.values.sum := CliqueSearch.member_le_sum tree.values
  have optimal := (CliqueSearch.search_optimal relation need [] tree.values 0 before ordered).change_relation_on
    ordered agreement bounded
  exact optimal.unique (CliqueSearch.search_optimal graph.edge need [] tree.values 0 before ordered)

end ProjectEulerSolutions.PrimePairCertificate
