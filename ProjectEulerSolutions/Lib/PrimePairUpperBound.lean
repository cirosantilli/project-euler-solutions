import ProjectEulerSolutions.Lib.PrimePairFast
import ProjectEulerSolutions.Lib.CliqueSupergraph

namespace ProjectEulerSolutions.PrimePairCertificate

/-- Included neighbors need no numeric proof. For an excluded neighbor,
check one producer-selected concatenation that the source must reject. -/
def rejectionCheck (p q : Nat) (neighbors rejectSecond : List Nat) : Bool :=
  if p < q then
    if neighbors.contains q then true else
      if (p + q) % 3 = 0 then true else
        if rejectSecond.contains q then !MillerRabin.Fast.model (q * DecimalShift.fiveDigits p + p)
        else !MillerRabin.Fast.model (p * DecimalShift.fiveDigits q + q)
  else true

def rejectsMissing : VertexTree → Nat → List Nat → List Nat → Bool
  | .leaf values, p, neighbors, rejectSecond => values.all (fun q => rejectionCheck p q neighbors rejectSecond)
  | .branch left right, p, neighbors, rejectSecond =>
      rejectsMissing left p neighbors rejectSecond && rejectsMissing right p neighbors rejectSecond

theorem rejectionCheck_sound (p q : Nat) (neighbors rejectSecond : List Nat)
    (checked : rejectionCheck p q neighbors rejectSecond = true) (ordered : p < q)
    (accepted : relation p q = true) : q ∈ neighbors := by
  have notThree : ¬ (p + q) % 3 = 0 := by
    intro divisible
    simp [relation, divisible] at accepted
  have both : MillerRabin.isPrimeModel (p * DecimalShift.fiveDigits q + q) = true ∧
      MillerRabin.isPrimeModel (q * DecimalShift.fiveDigits p + p) = true := by
    simpa only [relation, if_neg notThree, Bool.and_eq_true] using accepted
  simpa [rejectionCheck, ordered, notThree, MillerRabin.Fast.model_eq_source, both.1, both.2] using checked

theorem rejectsMissing_at (tree : VertexTree) (p q : Nat) (neighbors rejectSecond : List Nat)
    (checked : rejectsMissing tree p neighbors rejectSecond = true) (member : q ∈ tree.values) :
    rejectionCheck p q neighbors rejectSecond = true := by
  induction tree with
  | leaf values => exact List.all_eq_true.mp checked q member
  | branch left right ihLeft ihRight =>
    simp only [rejectsMissing, Bool.and_eq_true] at checked
    rcases List.mem_append.mp member with inLeft | inRight
    · exact ihLeft checked.1 inLeft
    · exact ihRight checked.2 inRight

def RowUpperBound (tree : VertexTree) (p : Nat) (neighbors : List Nat) : Prop :=
  ∀ q ∈ tree.values, p < q → relation p q = true → q ∈ neighbors

theorem row_upper_of_checked (tree : VertexTree) (p : Nat) (neighbors rejectSecond : List Nat)
    (checked : rejectsMissing tree p neighbors rejectSecond = true) : RowUpperBound tree p neighbors := by
  intro q member ordered accepted
  exact rejectionCheck_sound p q neighbors rejectSecond
    (rejectsMissing_at tree p q neighbors rejectSecond checked member) ordered accepted

/-- The numeric cache may overapproximate accepted edges. -/
def Graph.Covers (tree : VertexTree) (graph : Graph) : Prop :=
  ∀ p ∈ graph.vertices, ∀ q ∈ tree.values, p < q → relation p q = true → graph.edge p q = true

theorem Graph.Agrees.covers {graph : Graph} {tree : VertexTree} (agrees : graph.Agrees tree) : graph.Covers tree := by
  intro p memberP q memberQ ordered accepted
  rw [agrees p memberP q memberQ ordered]
  exact accepted

theorem Graph.empty_covers (tree : VertexTree) : Graph.Covers tree .empty := by
  intro p member
  simp [Graph.vertices] at member

theorem Graph.node_covers (tree : VertexTree) (p : Nat) (neighbors : List Nat) (left right : Graph)
    (leftCovers : left.Covers tree) (rightCovers : right.Covers tree)
    (before : ∀ value ∈ left.vertices, value < p) (after : ∀ value ∈ right.vertices, p < value)
    (row : RowUpperBound tree p neighbors) : Graph.Covers tree (.node p neighbors left right) := by
  intro query member q memberQ ordered accepted
  simp only [Graph.vertices, List.mem_append, List.mem_cons] at member
  rcases member with inLeft | same | inRight
  · simp only [Graph.edge, before query inLeft, ↓reduceIte]
    exact leftCovers query inLeft q memberQ ordered accepted
  · subst query
    simpa only [Graph.edge, Nat.lt_irrefl, ↓reduceIte, List.contains_iff_mem] using row q memberQ ordered accepted
  · simp only [Graph.edge, show ¬ query < p by have := after query inRight; omega, after query inRight, ↓reduceIte]
    exact rightCovers query inRight q memberQ ordered accepted

theorem Graph.included {graph : Graph} {tree : VertexTree} (covers : graph.Covers tree)
    (coverage : ∀ p ∈ tree.values, p ∈ graph.vertices) :
    CliqueSearch.IncludedOn relation graph.edge tree.values := by
  intro p memberP q memberQ ordered accepted
  exact covers p (coverage p memberP) q memberQ ordered accepted

end ProjectEulerSolutions.PrimePairCertificate
