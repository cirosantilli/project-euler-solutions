import Mathlib.Data.List.MinMax
import Mathlib.Data.List.Pairwise
import Mathlib.Data.List.Range
import Mathlib.Data.Nat.Prime.Basic
import Lean.Elab.Tactic.Omega

/-! Prefix sums, monotone windows, and first-on-tie selection for consecutive-prime searches. -/
namespace ProjectEulerSolutions.ConsecutivePrimeSums

set_option Elab.async false

def fromTotal (total : Nat) : List Nat → List Nat
  | [] => []
  | head :: tail => (total + head) :: fromTotal (total + head) tail

def prefixes (items : List Nat) : List Nat := 0 :: fromTotal 0 items
def window (items : List Nat) (first length : Nat) : Nat := ((items.drop first).take length).sum

@[simp] theorem fromTotal_length (items : List Nat) (total : Nat) :
    (fromTotal total items).length = items.length := by
  induction items generalizing total with
  | nil => rfl
  | cons head tail ih => simp [fromTotal, ih]

@[simp] theorem prefix_length (items : List Nat) : (prefixes items).length = items.length + 1 := by
  simp [prefixes]

theorem fromTotal_append (front back : List Nat) (total : Nat) :
    fromTotal total (front ++ back) = fromTotal total front ++ fromTotal (total + front.sum) back := by
  induction front generalizing total with
  | nil => simp [fromTotal]
  | cons head tail ih => simp [fromTotal, ih, Nat.add_assoc]

theorem prefix_append (items : List Nat) (value : Nat) :
    prefixes (items ++ [value]) = prefixes items ++ [items.sum + value] := by
  simp [prefixes, fromTotal_append, fromTotal]

theorem fromTotal_get (items : List Nat) (total position : Nat) (inside : position < items.length) :
    (fromTotal total items)[position]? = some (total + (items.take (position + 1)).sum) := by
  induction items generalizing total position with
  | nil => simp at inside
  | cons head tail ih =>
      cases position with
      | zero => simp [fromTotal]
      | succ position => simpa [fromTotal, Nat.add_assoc] using ih (total + head) position (by simpa using inside)

theorem prefix_get (items : List Nat) (position : Nat) (inside : position ≤ items.length) :
    (prefixes items)[position]? = some ((items.take position).sum) := by
  cases position with
  | zero => simp [prefixes]
  | succ position =>
      simpa only [prefixes, List.getElem?_cons_succ, Nat.zero_add] using fromTotal_get items 0 position (by omega)

theorem window_succ (items : List Nat) (first length : Nat) (inside : first < items.length) :
    window items first (length + 1) = items[first] + window items (first + 1) length := by
  unfold window
  rw [List.drop_eq_getElem_cons inside, List.take_succ_cons, List.sum_cons]

/-- The two prefix entries encode the exact half-open interval, without truncated subtraction. -/
theorem prefix_difference (items : List Nat) (first length : Nat) (inside : first + length ≤ items.length) :
    ((prefixes items)[first + length]?).getD 0 - ((prefixes items)[first]?).getD 0 = window items first length := by
  rw [prefix_get items (first + length) inside, prefix_get items first (by omega)]
  simp only [Option.getD_some]
  have split := congrArg List.sum (List.take_append_drop first (items.take (first + length)))
  simp only [List.sum_append, List.take_take, Nat.min_eq_left (show first ≤ first + length by omega),
    List.drop_take, Nat.add_sub_cancel_left] at split
  unfold window
  omega

theorem window_mono_length (items : List Nat) (first shorter longer : Nat) (ordered : shorter ≤ longer) :
    window items first shorter ≤ window items first longer := by
  have split := congrArg List.sum (List.take_append_drop shorter ((items.drop first).take longer))
  simp only [List.sum_append, List.take_take, Nat.min_eq_left ordered] at split
  unfold window
  omega

/-- Increasing the start of an in-bounds fixed-width window cannot lower its sum. -/
theorem window_mono_start (items : List Nat) (ordered : items.Pairwise (· ≤ ·))
    (first later length : Nat) (starts : first ≤ later) (inside : later + length ≤ items.length) :
    window items first length ≤ window items later length := by
  induction length generalizing first later with
  | zero => simp [window]
  | succ length ih =>
      have firstInside : first < items.length := by omega
      have laterInside : later < items.length := by omega
      rw [window_succ items first length firstInside, window_succ items later length laterInside]
      apply Nat.add_le_add
      · exact ordered.rel_get_of_le (a := ⟨first, firstInside⟩) (b := ⟨later, laterInside⟩) starts
      · exact ih (first + 1) (later + 1) (by omega) (by omega)

structure Window where
  first : Nat
  length : Nat
  deriving DecidableEq, Repr

def Valid (items : List Nat) (limit : Nat) (candidate : Window) : Prop :=
  0 < candidate.length ∧ candidate.first + candidate.length ≤ items.length ∧
    window items candidate.first candidate.length < limit ∧ Nat.Prime (window items candidate.first candidate.length)

instance (items : List Nat) (limit : Nat) (candidate : Window) : Decidable (Valid items limit candidate) := by
  unfold Valid
  infer_instance

def Best (items : List Nat) (limit : Nat) (candidate : Window) : Prop :=
  Valid items limit candidate ∧ ∀ other, Valid items limit other →
    other.length ≤ candidate.length ∧ (other.length = candidate.length → candidate.first ≤ other.first)

theorem best_unique {items : List Nat} {limit : Nat} {left right : Window}
    (leftBest : Best items limit left) (rightBest : Best items limit right) : left = right := by
  obtain ⟨leftLength, leftFirst⟩ := leftBest.2 right rightBest.1
  obtain ⟨rightLength, rightFirst⟩ := rightBest.2 left leftBest.1
  have sameLength : left.length = right.length := by omega
  have sameFirst : left.first = right.first := Nat.le_antisymm (leftFirst sameLength.symm) (rightFirst sameLength)
  cases left
  cases right
  simp_all

/-- Once the first window at the next length is too large, every longer valid window is excluded. -/
theorem maximum_length_bound (items : List Nat) (ordered : items.Pairwise (· ≤ ·)) (limit maximum : Nat)
    (stopped : maximum = items.length ∨ limit ≤ window items 0 (maximum + 1))
    (candidate : Window) (valid : Valid items limit candidate) : candidate.length ≤ maximum := by
  by_contra longer
  have length : maximum + 1 ≤ candidate.length := by omega
  rcases stopped with stopped | stopped
  · have := valid.2.1
    omega
  · have enlarged := window_mono_length items 0 (maximum + 1) candidate.length length
    have moved := window_mono_start items ordered 0 candidate.first candidate.length (by omega) valid.2.1
    have := valid.2.2.1
    omega

def row (items : List Nat) (limit first : Nat) : List Window :=
  ((List.range (items.length + 1)).map (fun stop => Window.mk first (stop - first))).filter (fun candidate => decide (Valid items limit candidate))

def candidates (items : List Nat) (limit : Nat) : List Window :=
  (List.range (items.length + 1)).flatMap (row items limit)

theorem row_first {items : List Nat} {limit first : Nat} {candidate : Window}
    (member : candidate ∈ row items limit first) : candidate.first = first := by
  obtain ⟨listed, _⟩ := List.mem_filter.mp member
  obtain ⟨stop, _, rfl⟩ := List.mem_map.mp listed
  rfl

theorem mem_candidates (items : List Nat) (limit : Nat) (candidate : Window) :
    candidate ∈ candidates items limit ↔ Valid items limit candidate := by
  constructor
  · intro member
    obtain ⟨first, _, member⟩ := List.mem_flatMap.mp member
    exact of_decide_eq_true (List.mem_filter.mp member).2
  · intro valid
    have within : candidate.first < items.length + 1 := by have := valid.2.1; omega
    apply List.mem_flatMap.mpr
    refine ⟨candidate.first, List.mem_range.mpr within, ?_⟩
    apply List.mem_filter.mpr
    refine ⟨?_, by simp [valid]⟩
    apply List.mem_map.mpr
    refine ⟨candidate.first + candidate.length, List.mem_range.mpr (by have := valid.2.1; omega), ?_⟩
    cases candidate
    simp

theorem candidates_ordered (items : List Nat) (limit : Nat) :
    (candidates items limit).Pairwise (fun left right => left.first ≤ right.first) := by
  apply List.pairwise_flatMap.mpr
  constructor
  · intro first member
    apply List.pairwise_of_forall_mem_list
    intro left leftMember right rightMember
    simp only [row_first leftMember, row_first rightMember, le_refl]
  · apply List.pairwise_le_range.imp
    intro first later starts left leftMember right rightMember
    simpa only [row_first leftMember, row_first rightMember] using starts

def reference (items : List Nat) (limit : Nat) : Option Window := (candidates items limit).argmax Window.length

/-- The exhaustive enumeration chooses the longest window and then its earliest start. -/
theorem reference_best {items : List Nat} {limit : Nat} {chosen : Window}
    (selected : reference items limit = some chosen) : Best items limit chosen := by
  have member : chosen ∈ (candidates items limit).argmax Window.length := selected
  have chosenMember := List.argmax_mem member
  refine ⟨(mem_candidates items limit chosen).mp chosenMember, ?_⟩
  intro other valid
  have otherMember := (mem_candidates items limit other).mpr valid
  refine ⟨List.le_of_mem_argmax otherMember member, ?_⟩
  intro sameLength
  have positions := List.index_of_argmax member otherMember (by omega)
  have firstInside := List.idxOf_lt_length_of_mem chosenMember
  have otherInside := List.idxOf_lt_length_of_mem otherMember
  by_cases same : (candidates items limit).idxOf chosen = (candidates items limit).idxOf other
  · have equal : chosen = other := (List.idxOf_inj chosenMember).mp same
    simp [equal]
  · have ordered := (candidates_ordered items limit).rel_get_of_lt
      (a := ⟨_, firstInside⟩) (b := ⟨_, otherInside⟩)
      (by change (candidates items limit).idxOf chosen < (candidates items limit).idxOf other; omega)
    simpa only [List.get_eq_getElem, List.getElem_idxOf] using ordered

theorem reference_none (items : List Nat) (limit : Nat) :
    reference items limit = none ↔ ∀ candidate, ¬ Valid items limit candidate := by
  rw [reference, List.argmax_eq_none]
  constructor
  · intro empty candidate valid
    have member := (mem_candidates items limit candidate).mpr valid
    simp [empty] at member
  · intro empty
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro candidate member
    exact empty candidate ((mem_candidates items limit candidate).mp member)

def scanRow (items : List Nat) (limit length first : Nat) : Nat → Option Window
  | 0 => none
  | count + 1 =>
      if limit ≤ window items first length then none
      else if Nat.Prime (window items first length) then some ⟨first, length⟩
      else scanRow items limit length (first + 1) count

private def RowSpec (items : List Nat) (limit length first stop : Nat) : Option Window → Prop
  | none => ∀ start, first ≤ start → start < stop → ¬ Valid items limit ⟨start, length⟩
  | some chosen => Valid items limit chosen ∧ chosen.length = length ∧ first ≤ chosen.first ∧ chosen.first < stop ∧
      ∀ start, first ≤ start → start < stop → Valid items limit ⟨start, length⟩ → chosen.first ≤ start

/-- A row scan finds its earliest valid start; monotonicity justifies skipping the suffix after a break. -/
theorem scanRow_spec (items : List Nat) (ordered : items.Pairwise (· ≤ ·)) (limit length first count : Nat)
    (positive : 0 < length) (bounded : first + count + length ≤ items.length + 1) :
    RowSpec items limit length first (first + count) (scanRow items limit length first count) := by
  induction count generalizing first with
  | zero =>
      intro start lower upper
      omega
  | succ count ih =>
      by_cases tooLarge : limit ≤ window items first length
      · simp only [scanRow, if_pos tooLarge, RowSpec]
        intro start lower upper valid
        have moved := window_mono_start items ordered first start length lower valid.2.1
        have below : window items start length < limit := valid.2.2.1
        omega
      · by_cases prime : Nat.Prime (window items first length)
        · have valid : Valid items limit ⟨first, length⟩ := by
            change 0 < length ∧ first + length ≤ items.length ∧ window items first length < limit ∧
              Nat.Prime (window items first length)
            exact ⟨positive, by omega, by omega, prime⟩
          rw [scanRow, if_neg tooLarge, if_pos prime]
          refine ⟨valid, rfl, le_rfl, by change first < first + (count + 1); omega, ?_⟩
          intro start lower upper valid
          exact lower
        · have recursive := ih (first + 1) (by omega)
          cases answer : scanRow items limit length (first + 1) count with
          | none =>
              rw [answer] at recursive
              simp only [scanRow, if_neg tooLarge, if_neg prime, answer, RowSpec]
              intro start lower upper valid
              by_cases same : start = first
              · subst start
                exact prime valid.2.2.2
              · exact recursive start (by omega) (by omega) valid
          | some chosen =>
              rw [answer] at recursive
              obtain ⟨valid, sameLength, lower, upper, earliest⟩ := recursive
              simp only [scanRow, if_neg tooLarge, if_neg prime, answer, RowSpec]
              refine ⟨valid, sameLength, by omega, by omega, ?_⟩
              intro start startLower startUpper startValid
              by_cases same : start = first
              · subst start
                exact False.elim (prime startValid.2.2.2)
              · exact earliest start (by omega) (by omega) startValid

def scanLengths (items : List Nat) (limit : Nat) : Nat → Option Window
  | 0 => none
  | length + 1 =>
      (scanRow items limit (length + 1) 0 (items.length - (length + 1) + 1)).orElse
        (fun _ => scanLengths items limit length)

private def SearchSpec (items : List Nat) (limit maximum : Nat) : Option Window → Prop
  | none => ∀ candidate, Valid items limit candidate → ¬ candidate.length ≤ maximum
  | some chosen => Valid items limit chosen ∧ chosen.length ≤ maximum ∧
      ∀ other, Valid items limit other → other.length ≤ maximum →
        other.length ≤ chosen.length ∧ (other.length = chosen.length → chosen.first ≤ other.first)

theorem scanLengths_spec (items : List Nat) (ordered : items.Pairwise (· ≤ ·)) (limit maximum : Nat)
    (bounded : maximum ≤ items.length) :
    SearchSpec items limit maximum (scanLengths items limit maximum) := by
  induction maximum with
  | zero =>
      intro candidate valid
      have := valid.1
      omega
  | succ maximum ih =>
      have rows := scanRow_spec items ordered limit (maximum + 1) 0 (items.length - (maximum + 1) + 1)
        (by omega) (by omega)
      cases rowAnswer : scanRow items limit (maximum + 1) 0 (items.length - (maximum + 1) + 1) with
      | some chosen =>
          rw [rowAnswer] at rows
          obtain ⟨valid, sameLength, lower, upper, earliest⟩ := rows
          simp only [scanLengths, rowAnswer, Option.orElse_some, SearchSpec]
          refine ⟨valid, by omega, ?_⟩
          intro other otherValid otherBounded
          refine ⟨by omega, ?_⟩
          intro same
          have length : other.length = maximum + 1 := same.trans sameLength
          apply earliest other.first (by omega) (by have := otherValid.2.1; omega)
          simpa only [← length] using otherValid
      | none =>
          rw [rowAnswer] at rows
          have exclude (candidate : Window) (valid : Valid items limit candidate) : candidate.length ≠ maximum + 1 := by
            intro sameLength
            apply rows candidate.first (by omega) (by have := valid.2.1; omega)
            simpa only [← sameLength] using valid
          have recursive := ih (by omega)
          cases answer : scanLengths items limit maximum with
          | none =>
              rw [answer] at recursive
              simp only [scanLengths, rowAnswer, Option.orElse_none, answer, SearchSpec]
              intro candidate valid candidateBounded
              exact recursive candidate valid (by have := exclude candidate valid; omega)
          | some chosen =>
              rw [answer] at recursive
              obtain ⟨valid, length, greatest⟩ := recursive
              simp only [scanLengths, rowAnswer, Option.orElse_none, answer, SearchSpec]
              refine ⟨valid, by omega, ?_⟩
              intro other otherValid otherBounded
              exact greatest other otherValid (by have := exclude other otherValid; omega)

/-- The optimized search and exhaustive first-on-tie maximum select the identical interval. -/
theorem scanLengths_eq_reference (items : List Nat) (ordered : items.Pairwise (· ≤ ·)) (limit maximum : Nat)
    (bounded : maximum ≤ items.length)
    (covers : ∀ candidate, Valid items limit candidate → candidate.length ≤ maximum) :
    scanLengths items limit maximum = reference items limit := by
  have searched := scanLengths_spec items ordered limit maximum bounded
  cases answer : scanLengths items limit maximum with
  | none =>
      rw [answer] at searched
      symm
      apply (reference_none items limit).mpr
      intro candidate valid
      exact searched candidate valid (covers candidate valid)
  | some chosen =>
      rw [answer] at searched
      obtain ⟨valid, length, greatest⟩ := searched
      have chosenBest : Best items limit chosen := ⟨valid, fun other otherValid => greatest other otherValid (covers other otherValid)⟩
      cases selected : reference items limit with
      | none => exact False.elim (((reference_none items limit).mp selected chosen) valid)
      | some other =>
          have same := best_unique chosenBest (reference_best selected)
          simp only [same]

end ProjectEulerSolutions.ConsecutivePrimeSums
