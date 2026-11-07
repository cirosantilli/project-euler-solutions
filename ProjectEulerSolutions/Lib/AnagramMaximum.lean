import ProjectEulerSolutions.Lib.AnagramSearch

/-! Extensional bounds for the exhaustive anagram-square maximum. -/
namespace ProjectEulerSolutions.AnagramMaximum
open AnagramSquares AnagramSearch
set_option Elab.async false

theorem fold_bound {α : Type} (step : Int → α → Int) (property : α → Prop) (bound : Int)
    (stepBound : ∀ best item, step best item ≤ bound ↔ best ≤ bound ∧ property item)
    (items : List α) (best : Int) :
    items.foldl step best ≤ bound ↔ best ≤ bound ∧ ∀ item ∈ items, property item := by
  induction items generalizing best with
  | nil => simp
  | cons item rest ih => simp [List.foldl_cons, ih, stepBound, and_assoc]

def Accepted (word other : String) (root : Nat) : Prop :=
  match other.toList with
  | [] => False
  | first :: _ =>
      labels [] word.toList = labels [] (squareText root).toList ∧
      substitution '0' word.toList (squareText root).toList first ≠ '0' ∧
      (transformed word.toList (squareText root).toList other.toList).sqrt ^ 2 =
        transformed word.toList (squareText root).toList other.toList

def Bounded (word other : String) (root : Nat) (bound : Int) : Prop :=
  Accepted word other root →
    (root * root : Int) ≤ bound ∧ (transformed word.toList (squareText root).toList other.toList : Int) ≤ bound

theorem step_bound (word other : String) (root : Nat) (best bound : Int) :
    AnagramSearch.step word other root best ≤ bound ↔ best ≤ bound ∧ Bounded word other root bound := by
  cases view : other.toList with
  | nil => simp [AnagramSearch.step, Bounded, Accepted, view]
  | cons first rest =>
      by_cases matched : labels [] word.toList = labels [] (squareText root).toList
      · by_cases zero : substitution '0' word.toList (squareText root).toList first = '0'
        · simp [AnagramSearch.step, Bounded, Accepted, view, matched, zero]
        · by_cases square : (transformed word.toList (squareText root).toList other.toList).sqrt ^ 2 =
              transformed word.toList (squareText root).toList other.toList
          · simp_all [AnagramSearch.step, Bounded, Accepted]
          · simp_all [AnagramSearch.step, Bounded, Accepted]
      · simp [AnagramSearch.step, Bounded, Accepted, view, matched]

theorem roots_as_fold (word other : String) (first count : Nat) (best : Int) :
    scanRoots word other first count best =
      (List.range' first count).foldl (fun current root => AnagramSearch.step word other root current) best := by
  induction count generalizing first best with
  | zero => rfl
  | succ count ih => simp [scanRoots, List.range'_succ, ih]

theorem roots_bound (word other : String) (first count : Nat) (best bound : Int) :
    scanRoots word other first count best ≤ bound ↔
      best ≤ bound ∧ ∀ root ∈ List.range' first count, Bounded word other root bound := by
  rw [roots_as_fold]
  exact fold_bound _ _ bound (fun best root => step_bound word other root best bound) _ best

def PairBound (word other : String) (bound : Int) : Prop :=
  ∀ root, firstRoot word.length ≤ root → root ≤ lastRoot word.length → Bounded word other root bound

theorem root_member (width root : Nat) :
    root ∈ List.range' (firstRoot width) (lastRoot width + 1 - firstRoot width) ↔
      firstRoot width ≤ root ∧ root ≤ lastRoot width := by
  have ordered := ordered_roots width
  simp only [List.mem_range'_1]
  omega

theorem pair_bound (word other : String) (best bound : Int) :
    scanRoots word other (firstRoot word.length) (lastRoot word.length + 1 - firstRoot word.length) best ≤ bound ↔
      best ≤ bound ∧ PairBound word other bound := by
  rw [roots_bound]
  simp only [PairBound, root_member, and_imp]

theorem pairs_bound (pairs : List AnagramPairs.Pair) (best bound : Int) :
    scanPairs pairs best ≤ bound ↔ best ≤ bound ∧ ∀ pair ∈ pairs, PairBound pair.1 pair.2 bound := by
  induction pairs generalizing best with
  | nil => simp [scanPairs]
  | cons pair rest ih =>
      obtain ⟨word, other⟩ := pair
      simp [scanPairs, ih, pair_bound, and_assoc]

theorem solve_bound (words : List String) (bound : Int) :
    AnagramSearch.solve words ≤ bound ↔ 0 ≤ bound ∧ ∀ pair ∈ AnagramPairs.pairs words, PairBound pair.1 pair.2 bound :=
  pairs_bound _ 0 bound

end ProjectEulerSolutions.AnagramMaximum
