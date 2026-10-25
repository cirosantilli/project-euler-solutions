import Mathlib.Data.List.Basic
import Mathlib.Data.Int.Basic

/-! A finite model of the source's ordered clique search. Both lower-bound
breaks and the strict incumbent update are retained; no primality fact is assumed. -/
namespace ProjectEulerSolutions.CliqueSearch

structure Best where
  bound : Int
  chosen : List Nat
deriving DecidableEq, Repr

def improve (clique : List Nat) (sum : Int) (best : Best) : Best :=
  if sum < best.bound then ⟨sum, clique⟩ else best

def candidates (relation : Nat → Nat → Bool) (need : Nat) (sum : Int) (q : Nat)
    (remaining : List Nat) (bound : Int) : List Nat :=
  (remaining.takeWhile (fun (r : Nat) => decide (sum + (q : Int) + (r : Int) * ((need : Int) - 1) < bound))).filter (relation q)

def scan (relation : Nat → Nat → Bool) (recurse : List Nat → List Nat → Int → Best → Best)
    (need : Nat) (clique : List Nat) (sum : Int) : List Nat → Best → Best
  | [], best => best
  | q :: rest, best =>
      if best.bound ≤ sum + (q : Int) * need then best
      else
        let next := candidates relation need sum q rest best.bound
        let updated := if need - 1 ≤ next.length then recurse (clique ++ [q]) next (sum + q) best else best
        scan relation recurse need clique sum rest updated

def search (relation : Nat → Nat → Bool) : Nat → List Nat → List Nat → Int → Best → Best
  | 0, clique, _, sum, best => improve clique sum best
  | need + 1, clique, remaining, sum, best =>
      if remaining.length < need + 1 then best
      else scan relation (search relation need) (need + 1) clique sum remaining best

theorem improve_bound (clique : List Nat) (sum : Int) (best : Best) :
    (improve clique sum best).bound ≤ best.bound := by
  unfold improve
  split <;> simp_all
  omega

theorem improve_equal_bound (clique : List Nat) (sum : Int) (best : Best)
    (same : (improve clique sum best).bound = best.bound) : improve clique sum best = best := by
  unfold improve at same ⊢
  by_cases better : sum < best.bound
  · simp only [if_pos better] at same ⊢
    have equal : sum = best.bound := same
    omega
  · simp only [if_neg better]

theorem scan_bound (relation : Nat → Nat → Bool) (recurse : List Nat → List Nat → Int → Best → Best)
    (decreases : ∀ clique remaining sum best, (recurse clique remaining sum best).bound ≤ best.bound)
    (need : Nat) (clique : List Nat) (sum : Int) (remaining : List Nat) (best : Best) :
    (scan relation recurse need clique sum remaining best).bound ≤ best.bound := by
  induction remaining generalizing best with
  | nil => exact Int.le_refl _
  | cons q rest ih =>
      simp only [scan]
      split
      · exact Int.le_refl _
      · apply Int.le_trans (ih _)
        split
        · exact decreases _ _ _ _
        · exact Int.le_refl _

theorem search_bound (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat)
    (sum : Int) (best : Best) : (search relation need clique remaining sum best).bound ≤ best.bound := by
  induction need generalizing clique remaining sum best with
  | zero => exact improve_bound _ _ _
  | succ need ih =>
      simp only [search]
      split
      · exact Int.le_refl _
      · exact scan_bound relation (search relation need) (fun _ _ _ _ => ih _ _ _ _) _ _ _ _ _

theorem scan_equal_bound (relation : Nat → Nat → Bool) (recurse : List Nat → List Nat → Int → Best → Best)
    (decreases : ∀ clique remaining sum best, (recurse clique remaining sum best).bound ≤ best.bound)
    (stable : ∀ clique remaining sum best,
      (recurse clique remaining sum best).bound = best.bound → recurse clique remaining sum best = best)
    (need : Nat) (clique : List Nat) (sum : Int) (remaining : List Nat) (best : Best)
    (same : (scan relation recurse need clique sum remaining best).bound = best.bound) :
    scan relation recurse need clique sum remaining best = best := by
  induction remaining generalizing best with
  | nil => rfl
  | cons q rest ih =>
      simp only [scan] at same ⊢
      by_cases cut : best.bound ≤ sum + (q : Int) * need
      · simp only [if_pos cut]
      · simp only [if_neg cut] at same ⊢
        let next := candidates relation need sum q rest best.bound
        let updated := if need - 1 ≤ next.length then recurse (clique ++ [q]) next (sum + q) best else best
        have reduced : updated.bound ≤ best.bound := by
          unfold updated
          split
          · exact decreases _ _ _ _
          · exact Int.le_refl _
        change (scan relation recurse need clique sum rest updated).bound = best.bound at same
        have finalReduced := scan_bound relation recurse decreases need clique sum rest updated
        have updatedBound : updated.bound = best.bound := by
          omega
        have unchanged : updated = best := by
          unfold updated at updatedBound ⊢
          by_cases enough : need - 1 ≤ next.length
          · simp only [if_pos enough] at updatedBound ⊢
            exact stable _ _ _ _ updatedBound
          · simp only [if_neg enough]
        have tailBound : (scan relation recurse need clique sum rest updated).bound = updated.bound := by
          omega
        exact (ih updated tailBound).trans unchanged

theorem search_equal_bound (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat)
    (sum : Int) (best : Best) (same : (search relation need clique remaining sum best).bound = best.bound) :
    search relation need clique remaining sum best = best := by
  induction need generalizing clique remaining sum best with
  | zero => exact improve_equal_bound _ _ _ same
  | succ need ih =>
      simp only [search] at same ⊢
      by_cases short : remaining.length < need + 1
      · simp only [if_pos short]
      · simp only [if_neg short] at same ⊢
        exact scan_equal_bound relation (search relation need) (search_bound relation need)
          (fun _ _ _ _ same => ih _ _ _ _ same) _ _ _ _ _ same

theorem search_insufficient (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat)
    (sum : Int) (best : Best) (positive : 0 < need) (short : remaining.length < need) :
    search relation need clique remaining sum best = best := by
  cases need with
  | zero => omega
  | succ need => simp [search, short]

theorem search_head_pruned (relation : Nat → Nat → Bool) (need : Nat) (clique : List Nat)
    (q : Nat) (remaining : List Nat) (sum : Int) (best : Best) (positive : 0 < need)
    (enough : need ≤ (q :: remaining).length) (cut : best.bound ≤ sum + (q : Int) * need) :
    search relation need clique (q :: remaining) sum best = best := by
  cases need with
  | zero => omega
  | succ need => rw [search, if_neg (Nat.not_lt.mpr enough), scan, if_pos cut]

theorem candidates_sublist (relation : Nat → Nat → Bool) (need : Nat) (sum : Int) (q : Nat)
    (remaining : List Nat) (bound : Int) :
    List.Sublist (candidates relation need sum q remaining bound) remaining :=
  List.filter_sublist.trans (List.takeWhile_sublist _)

theorem candidates_compatible (relation : Nat → Nat → Bool) (need : Nat) (sum : Int) (q : Nat)
    (remaining : List Nat) (bound : Int) (r : Nat)
    (member : r ∈ candidates relation need sum q remaining bound) : relation q r = true :=
  (List.mem_filter.mp member).2

theorem sum_lower_bound (values : List Nat) (lower : Nat) (bounded : ∀ value ∈ values, lower ≤ value) :
    lower * values.length ≤ values.sum := by
  induction values with
  | nil => simp
  | cons value rest ih =>
      have head := bounded value (by simp)
      have tail := ih (fun item member => bounded item (by simp [member]))
      simpa [Nat.mul_add, Nat.add_comm] using Nat.add_le_add head tail

/-- The same inequality justifies both source breaks: first for every remaining
pick, then for the picks still needed after selecting the current vertex. -/
theorem bound_prunes (values : List Nat) (count lower : Nat) (base bound : Int)
    (size : values.length = count) (bounded : ∀ value ∈ values, lower ≤ value)
    (cut : bound ≤ base + (lower : Int) * count) : bound ≤ base + (values.sum : Int) := by
  have natural := sum_lower_bound values lower bounded
  rw [size] at natural
  have integer : (lower : Int) * count ≤ (values.sum : Int) := by exact_mod_cast natural
  omega

end ProjectEulerSolutions.CliqueSearch
