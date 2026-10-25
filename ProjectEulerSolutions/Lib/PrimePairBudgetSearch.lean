import ProjectEulerSolutions.Lib.PrimePairBudget
import Mathlib.Data.List.TakeWhile

namespace ProjectEulerSolutions.PrimePairCertificate.Budget
open CliqueSearch

/-- The source's candidate cutoff excludes every pair relaxed by the budget. -/
theorem candidates_eq_lookup (graph : Graph) (bound need : Nat) (sum : Int) (p : Nat)
    (remaining : List Nat) (bestBound : Int) (nonnegative : 0 ≤ sum) (bounded : bestBound ≤ (bound : Int)) :
    candidates (possible graph bound) need sum p remaining bestBound =
      candidates graph.lookupEdge need sum p remaining bestBound := by
  unfold candidates
  apply List.filter_congr
  intro q member
  have inside : sum + (p : Int) + (q : Int) < bestBound := by
    simpa only [decide_eq_true_eq] using (List.mem_takeWhile_imp member)
  have competitive : ¬ bound ≤ p + q := by omega
  simp only [possible, if_neg competitive]

private theorem scan_eq_lookup (graph : Graph) (bound need : Nat)
    (child : ∀ (clique remaining : List Nat) (sum : Int) (best : Best), 0 ≤ sum → best.bound ≤ (bound : Int) →
      search (possible graph bound) need clique remaining sum best =
        search graph.lookupEdge need clique remaining sum best)
    (clique remaining : List Nat) (sum : Int) (best : Best)
    (nonnegative : 0 ≤ sum) (bounded : best.bound ≤ (bound : Int)) :
    scan (possible graph bound) (search (possible graph bound) need) (need + 1) clique sum remaining best =
      scan graph.lookupEdge (search graph.lookupEdge need) (need + 1) clique sum remaining best := by
  induction remaining generalizing best with
  | nil => rfl
  | cons p rest ih =>
    rw [scan, scan]
    by_cases cut : best.bound ≤ sum + (p : Int) * ((need + 1 : Nat) : Int)
    · simp only [if_pos cut]
    · rw [if_neg cut, if_neg cut, candidates_eq_lookup graph bound (need + 1) sum p rest best.bound nonnegative bounded]
      let next := candidates graph.lookupEdge (need + 1) sum p rest best.bound
      by_cases enough : need + 1 - 1 ≤ next.length
      · have same := child (clique ++ [p]) next (sum + p) best (by omega) bounded
        change scan (possible graph bound) (search (possible graph bound) need) (need + 1) clique sum rest
            (if need + 1 - 1 ≤ next.length then search (possible graph bound) need (clique ++ [p]) next (sum + p) best else best) =
          scan graph.lookupEdge (search graph.lookupEdge need) (need + 1) clique sum rest
            (if need + 1 - 1 ≤ next.length then search graph.lookupEdge need (clique ++ [p]) next (sum + p) best else best)
        rw [if_pos enough, if_pos enough, same]
        apply ih
        exact le_trans (search_bound graph.lookupEdge need (clique ++ [p]) next (sum + p) best) bounded
      · change scan (possible graph bound) (search (possible graph bound) need) (need + 1) clique sum rest
            (if need + 1 - 1 ≤ next.length then search (possible graph bound) need (clique ++ [p]) next (sum + p) best else best) =
          scan graph.lookupEdge (search graph.lookupEdge need) (need + 1) clique sum rest
            (if need + 1 - 1 ≤ next.length then search graph.lookupEdge need (clique ++ [p]) next (sum + p) best else best)
        rw [if_neg enough, if_neg enough]
        exact ih best bounded

/-- A seeded source search can use the sparse lookup relation directly.
The equality preserves the complete incumbent, including its tie selection. -/
theorem search_eq_lookup (graph : Graph) (bound need : Nat) (clique remaining : List Nat)
    (sum : Int) (best : Best) (nonnegative : 0 ≤ sum) (bounded : best.bound ≤ (bound : Int)) :
    search (possible graph bound) need clique remaining sum best =
      search graph.lookupEdge need clique remaining sum best := by
  induction need generalizing clique remaining sum best with
  | zero => rfl
  | succ need ih =>
    change (if remaining.length < need + 1 then best else
      scan (possible graph bound) (search (possible graph bound) need) (need + 1) clique sum remaining best) =
      (if remaining.length < need + 1 then best else
        scan graph.lookupEdge (search graph.lookupEdge need) (need + 1) clique sum remaining best)
    by_cases short : remaining.length < need + 1
    · simp only [if_pos short]
    · simp only [if_neg short]
      exact scan_eq_lookup graph bound need ih clique remaining sum best nonnegative bounded

/-- Reuse a fully checked small-domain row inside the larger sum budget. -/
theorem row_upper_of_exact (tree : VertexTree) (bound p : Nat) (neighbors : List Nat)
    (checked : PrimePairCertificate.upperNeighbors tree p = neighbors) : RowUpperBound tree bound p neighbors := by
  intro q member ordered _ accepted
  rw [← checked]
  exact (mem_upperNeighbors tree p q).mpr ⟨member, ordered, accepted⟩

theorem empty_row_before (tree : VertexTree) (bound p upper : Nat)
    (values : ∀ q ∈ tree.values, q < upper) (past : upper ≤ p) : RowUpperBound tree bound p [] := by
  intro q member ordered _ _
  have := values q member
  omega

theorem empty_row_after (tree : VertexTree) (bound p lower : Nat)
    (values : ∀ q ∈ tree.values, lower ≤ q) (outside : bound ≤ p + lower) : RowUpperBound tree bound p [] := by
  intro q member _ within _
  have := values q member
  omega


end ProjectEulerSolutions.PrimePairCertificate.Budget
