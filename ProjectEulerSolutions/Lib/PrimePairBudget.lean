import ProjectEulerSolutions.Lib.PrimePairUpperBound
import ProjectEulerSolutions.Lib.PrimePairLookup
import ProjectEulerSolutions.Lib.MillerRabinFactor

namespace ProjectEulerSolutions.PrimePairCertificate.Budget

/-- Omitted pairs outside the improving sum budget need only an arithmetic check. -/
inductive Choice where
  | outside
  | possible
  | first
  | second
  | firstFactor (divisor : Nat)
  | secondFactor (divisor : Nat)

private def checkFactor (n divisor : Nat) : Bool :=
  MillerRabin.smallPrimes.contains divisor && (decide (37 < n) && decide (n % divisor = 0))

private theorem checkFactor_sound {n divisor : Nat} (checked : checkFactor n divisor = true) : MillerRabin.Fast.model n = false := by
  simp only [checkFactor, Bool.and_eq_true, List.contains_iff_mem, decide_eq_true_eq] at checked
  exact MillerRabin.smallFactor_rejects n divisor checked.2.1 checked.1 checked.2.2

private def Choice.check (choice : Choice) (bound p q : Nat) : Bool :=
  match choice with
  | .outside => decide (q ≤ p ∨ bound ≤ p + q)
  | .possible => true
  | .first => rejectionCheck p q [] []
  | .second => rejectionCheck p q [] [q]
  | .firstFactor divisor => checkFactor (p * DecimalShift.fiveDigits q + q) divisor
  | .secondFactor divisor => checkFactor (q * DecimalShift.fiveDigits p + p) divisor

private theorem Choice.possible_of_checked (choice : Choice) (bound p q : Nat)
    (checked : choice.check bound p q = true) (ordered : p < q) (within : p + q < bound)
    (accepted : relation p q = true) : choice = .possible := by
  cases choice with
  | outside =>
    have outside : q ≤ p ∨ bound ≤ p + q := by simpa only [Choice.check, decide_eq_true_eq] using checked
    omega
  | possible => rfl
  | first => exact False.elim (by simpa using rejectionCheck_sound p q [] [] checked ordered accepted)
  | second => exact False.elim (by simpa using rejectionCheck_sound p q [] [q] checked ordered accepted)
  | firstFactor divisor =>
    have failed := checkFactor_sound checked
    have rejected : rejectionCheck p q [] [] = true := by simp [rejectionCheck, ordered, failed]
    exact False.elim (by simpa using rejectionCheck_sound p q [] [] rejected ordered accepted)
  | secondFactor divisor =>
    have failed := checkFactor_sound checked
    have rejected : rejectionCheck p q [] [q] = true := by simp [rejectionCheck, ordered, failed]
    exact False.elim (by simpa using rejectionCheck_sound p q [] [q] rejected ordered accepted)

def checkPlan (bound p : Nat) : List Nat → List Choice → Bool
  | [], [] => true
  | q :: qs, choice :: choices => choice.check bound p q && checkPlan bound p qs choices
  | _, _ => false

def planNeighbors : List Nat → List Choice → List Nat
  | q :: qs, .possible :: choices => q :: planNeighbors qs choices
  | _ :: qs, _ :: choices => planNeighbors qs choices
  | _, _ => []

def RowUpperBound (tree : VertexTree) (bound p : Nat) (neighbors : List Nat) : Prop :=
  ∀ q ∈ tree.values, p < q → p + q < bound → relation p q = true → q ∈ neighbors

theorem plan_covers (bound p : Nat) (values : List Nat) (choices : List Choice)
    (checked : checkPlan bound p values choices = true) :
    ∀ q ∈ values, p < q → p + q < bound → relation p q = true → q ∈ planNeighbors values choices := by
  induction values generalizing choices with
  | nil => simp
  | cons head rest ih =>
    cases choices with
    | nil => simp [checkPlan] at checked
    | cons choice choices =>
      simp only [checkPlan, Bool.and_eq_true] at checked
      intro q member ordered within accepted
      rcases List.mem_cons.mp member with same | member
      · subst q
        have same := Choice.possible_of_checked choice bound p head checked.1 ordered within accepted
        simp [same, planNeighbors]
      · have found := ih choices checked.2 q member ordered within accepted
        cases choice <;> simp [planNeighbors, found]

theorem row_upper_of_plan (tree : VertexTree) (bound p : Nat) (choices : List Choice) (neighbors : List Nat)
    (checked : checkPlan bound p tree.values choices = true) (listed : planNeighbors tree.values choices = neighbors) :
    RowUpperBound tree bound p neighbors := by
  rw [← listed]
  exact plan_covers bound p tree.values choices checked

theorem RowUpperBound.branch {left right : VertexTree} {bound p : Nat} {leftRow rightRow : List Nat}
    (leftBound : RowUpperBound left bound p leftRow) (rightBound : RowUpperBound right bound p rightRow) :
    RowUpperBound (.branch left right) bound p (leftRow ++ rightRow) := by
  intro q member ordered within accepted
  rcases List.mem_append.mp member with inLeft | inRight
  · exact List.mem_append.mpr (Or.inl (leftBound q inLeft ordered within accepted))
  · exact List.mem_append.mpr (Or.inr (rightBound q inRight ordered within accepted))

theorem RowUpperBound.of_unbounded {tree : VertexTree} {bound p : Nat} {neighbors : List Nat} (row : PrimePairCertificate.RowUpperBound tree p neighbors) :
    RowUpperBound tree bound p neighbors := by
  intro q member ordered _ accepted
  exact row q member ordered accepted

/-- Sparse rows cover competitive pairs; the completed relation below retains
all uncompetitive pairs as possible without storing them in the graph. -/
def Covers (tree : VertexTree) (bound : Nat) (graph : Graph) : Prop :=
  ∀ p ∈ graph.vertices, ∀ q ∈ tree.values, p < q → p + q < bound →
    relation p q = true → graph.edge p q = true

theorem empty_covers (tree : VertexTree) (bound : Nat) : Covers tree bound .empty := by
  intro p member
  simp [Graph.vertices] at member

theorem node_covers (tree : VertexTree) (bound p : Nat) (neighbors : List Nat) (left right : Graph)
    (leftCovers : Covers tree bound left) (rightCovers : Covers tree bound right)
    (before : ∀ value ∈ left.vertices, value < p) (after : ∀ value ∈ right.vertices, p < value)
    (row : RowUpperBound tree bound p neighbors) : Covers tree bound (.node p neighbors left right) := by
  intro query member q memberQ ordered within accepted
  simp only [Graph.vertices, List.mem_append, List.mem_cons] at member
  rcases member with inLeft | same | inRight
  · simp only [Graph.edge, before query inLeft, ↓reduceIte]
    exact leftCovers query inLeft q memberQ ordered within accepted
  · subst query
    simpa only [Graph.edge, Nat.lt_irrefl, ↓reduceIte, List.contains_iff_mem] using row q memberQ ordered within accepted
  · simp only [Graph.edge, show ¬ query < p by have := after query inRight; omega, after query inRight, ↓reduceIte]
    exact rightCovers query inRight q memberQ ordered within accepted

/-- Expensive rejection proofs are needed only below the seeded sum budget.
Keeping every other pair possible preserves all actual completions and ties. -/
def possible (graph : Graph) (bound p q : Nat) : Bool :=
  if bound ≤ p + q then true else graph.lookupEdge p q

theorem included {tree : VertexTree} {bound : Nat} {graph : Graph} (covers : Covers tree bound graph)
    (coverage : ∀ p ∈ tree.values, 2 * p < bound → p ∈ graph.vertices) :
    CliqueSearch.IncludedOn relation (possible graph bound) tree.values := by
  intro p memberP q memberQ ordered accepted
  by_cases outside : bound ≤ p + q
  · simp only [possible, if_pos outside]
  · simp only [possible, if_neg outside]
    rw [← Graph.edge_eq_lookup_at]
    exact covers p (coverage p memberP (by omega)) q memberQ ordered (by omega) accepted

end ProjectEulerSolutions.PrimePairCertificate.Budget
