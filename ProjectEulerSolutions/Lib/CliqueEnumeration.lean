import ProjectEulerSolutions.Lib.CliqueSearchCorrectness

/-! Exhaustive ordered clique enumeration and strict-minimum selection.
The reference-style traversal and pruned search satisfy the same specification. -/
namespace ProjectEulerSolutions.CliqueEnumeration
open CliqueSearch

def accepts (relation : Nat → Nat → Bool) (front : List Nat) (q : Nat) : Bool :=
  ((front.getLast?.map (fun p => decide (p < q))).getD true) && front.all (fun p => relation p q)

def enumerate (relation : Nat → Nat → Bool) : Nat → List Nat → List Nat → List (List Nat)
  | 0, front, _ => [front]
  | need + 1, front, values => values.flatMap (fun q =>
      if accepts relation front q then enumerate relation need (front ++ [q]) values else [])

def select (options : List (List Nat)) (best : Best) : Best :=
  options.foldl (fun best chosen => improve chosen (chosen.sum : Int) best) best

def exhaustive (relation : Nat → Nat → Bool) (need : Nat) (front values : List Nat) (best : Best) : Best :=
  select (enumerate relation need front values) best

theorem exhaustive_zero (relation : Nat → Nat → Bool) (front values : List Nat) (best : Best) :
    exhaustive relation 0 front values best = improve front (front.sum : Int) best := rfl

theorem exhaustive_succ (relation : Nat → Nat → Bool) (need : Nat) (front values : List Nat) (best : Best) :
    exhaustive relation (need + 1) front values best = values.foldl (fun best q =>
      if accepts relation front q then exhaustive relation need (front ++ [q]) values best else best) best := by
  unfold exhaustive select
  rw [enumerate, List.foldl_flatMap]
  congr 1
  funext best q
  split <;> rfl

theorem last_guard_iff (front : List Nat) (q : Nat) (ordered : front.Pairwise (· < ·)) :
    (front.getLast?.map (fun p => decide (p < q))).getD true = true ↔ ∀ p ∈ front, p < q := by
  induction front with
  | nil => simp
  | cons first rest ih =>
    have sorted := List.pairwise_cons.mp ordered
    cases rest with
    | nil => simp
    | cons next tail =>
      rw [List.getLast?_cons_cons, ih sorted.2]
      constructor
      · intro bounded p member
        rcases List.mem_cons.mp member with same | member
        · subst p
          exact Nat.lt_trans (sorted.1 next (by simp)) (bounded next (by simp))
        · exact bounded p member
      · intro bounded p member
        exact bounded p (List.mem_cons_of_mem first member)

theorem accepts_iff (relation : Nat → Nat → Bool) (front : List Nat) (q : Nat)
    (ordered : front.Pairwise (· < ·)) : accepts relation front q = true ↔
    (∀ p ∈ front, p < q) ∧ (∀ p ∈ front, relation p q = true) := by
  simp only [accepts, Bool.and_eq_true, last_guard_iff front q ordered, List.all_eq_true]

def Valid (relation : Nat → Nat → Bool) (front : List Nat) : Prop :=
  front.Pairwise (· < ·) ∧ front.Pairwise (fun p q => relation p q = true)

theorem Valid.append {relation : Nat → Nat → Bool} {front : List Nat} {q : Nat}
    (valid : Valid relation front) (allowed : accepts relation front q = true) : Valid relation (front ++ [q]) := by
  have allowed := (accepts_iff relation front q valid.1).mp allowed
  constructor
  · apply List.pairwise_append.mpr
    refine ⟨valid.1, by simp, ?_⟩
    intro p member r singleton
    have same := List.mem_singleton.mp singleton
    simpa only [same] using allowed.1 p member
  · apply List.pairwise_append.mpr
    refine ⟨valid.2, by simp, ?_⟩
    intro p member r singleton
    have same := List.mem_singleton.mp singleton
    simpa only [same] using allowed.2 p member

theorem mem_enumerate (relation : Nat → Nat → Bool) (need : Nat) (front values chosen : List Nat)
    (valid : Valid relation front) : chosen ∈ enumerate relation need front values ↔
    ∃ tail, chosen = front ++ tail ∧ tail.length = need ∧ (∀ p ∈ tail, p ∈ values) ∧ Valid relation (front ++ tail) := by
  induction need generalizing front chosen with
  | zero =>
    constructor
    · intro member
      have same : chosen = front := by simpa only [enumerate, List.mem_singleton] using member
      exact ⟨[], by simpa using same, rfl, by simp, by simpa using valid⟩
    · rintro ⟨tail, same, size, _, _⟩
      have empty := List.length_eq_zero_iff.mp size
      simpa only [empty, List.append_nil, enumerate, List.mem_singleton] using same
  | succ need ih =>
    constructor
    · intro member
      obtain ⟨q, memberQ, child⟩ := List.mem_flatMap.mp member
      by_cases allowed : accepts relation front q = true
      · rw [if_pos allowed] at child
        obtain ⟨tail, same, size, members, childValid⟩ := (ih (front ++ [q]) chosen (valid.append allowed)).mp child
        refine ⟨q :: tail, by simpa [List.append_assoc] using same, by simp [size], ?_, ?_⟩
        · intro p member
          rcases List.mem_cons.mp member with same | member
          · simpa only [same] using memberQ
          · exact members p member
        · simpa only [List.append_assoc, List.singleton_append] using childValid
      · simp only [allowed, Bool.false_eq_true, if_false, List.not_mem_nil] at child
    · rintro ⟨tail, same, size, members, fullValid⟩
      cases tail with
      | nil => simp at size
      | cons q tail =>
        have allowed : accepts relation front q = true := by
          apply (accepts_iff relation front q valid.1).mpr
          have order := (List.pairwise_append.mp fullValid.1).2.2
          have pairs := (List.pairwise_append.mp fullValid.2).2.2
          exact ⟨fun p member => order p member q (by simp), fun p member => pairs p member q (by simp)⟩
        apply List.mem_flatMap.mpr
        refine ⟨q, members q (by simp), ?_⟩
        rw [if_pos allowed]
        apply (ih (front ++ [q]) chosen (valid.append allowed)).mpr
        refine ⟨tail, by simpa [List.append_assoc] using same, by simpa using size, ?_, ?_⟩
        · intro p member
          exact members p (List.mem_cons_of_mem q member)
        · simpa only [List.append_assoc, List.singleton_append] using fullValid

private theorem sorted_sublist (small large : List Nat) (smallSorted : small.Pairwise (· < ·))
    (largeSorted : large.Pairwise (· < ·)) (members : ∀ p ∈ small, p ∈ large) : small.Sublist large := by
  induction large generalizing small with
  | nil =>
    cases small with
    | nil => exact List.Sublist.refl _
    | cons q rest => have missing := members q (by simp); simp at missing
  | cons q rest ih =>
    have ordered := List.pairwise_cons.mp largeSorted
    cases small with
    | nil => exact List.nil_sublist _
    | cons r tail =>
      have firstOrder := List.pairwise_cons.mp smallSorted
      by_cases same : r = q
      · subst r
        apply List.Sublist.cons_cons
        apply ih tail firstOrder.2 ordered.2
        intro p member
        rcases List.mem_cons.mp (members p (List.mem_cons_of_mem q member)) with equal | inside
        · subst p
          have impossible := firstOrder.1 q member
          omega
        · exact inside
      · apply List.Sublist.cons
        apply ih (r :: tail) smallSorted ordered.2
        intro p member
        have rIn : r ∈ rest := (List.mem_cons.mp (members r (by simp))).resolve_left same
        have larger := ordered.1 r rIn
        rcases List.mem_cons.mp (members p member) with equal | inside
        · subst p
          rcases List.mem_cons.mp member with equal | inside
          · omega
          · have impossible := firstOrder.1 q inside
            omega
        · exact inside

theorem mem_enumerate_root (relation : Nat → Nat → Bool) (need : Nat) (values chosen : List Nat)
    (ordered : values.Pairwise (· < ·)) :
    chosen ∈ enumerate relation need [] values ↔ Completion relation need values chosen := by
  rw [mem_enumerate relation need [] values chosen ⟨by simp, by simp⟩]
  constructor
  · rintro ⟨tail, same, size, members, sorted, pairs⟩
    simp only [List.nil_append] at same sorted pairs
    subst chosen
    exact ⟨size, sorted_sublist tail values sorted ordered members, pairs⟩
  · intro completion
    exact ⟨chosen, rfl, completion.size, fun p member => completion.sublist.mem member,
      List.Pairwise.sublist completion.sublist ordered, completion.compatible⟩

private theorem lex_append_front (front : List Nat) {left right : List Nat}
    (less : List.Lex (· < ·) left right) : List.Lex (· < ·) (front ++ left) (front ++ right) := by
  induction front with
  | nil => exact less
  | cons head rest ih => exact List.Lex.cons ih

theorem enumerate_increasing (relation : Nat → Nat → Bool) (need : Nat) (front values : List Nat)
    (valid : Valid relation front) (ordered : values.Pairwise (· < ·)) :
    (enumerate relation need front values).Pairwise (List.Lex (· < ·)) := by
  induction need generalizing front with
  | zero => simp [enumerate]
  | succ need ih =>
    rw [enumerate, List.pairwise_flatMap]
    constructor
    · intro q member
      by_cases allowed : accepts relation front q = true
      · rw [if_pos allowed]
        exact ih _ (valid.append allowed)
      · simp [allowed]
    · apply ordered.imp
      intro q r less left memberLeft right memberRight
      by_cases leftAllowed : accepts relation front q = true
      · rw [if_pos leftAllowed] at memberLeft
        by_cases rightAllowed : accepts relation front r = true
        · rw [if_pos rightAllowed] at memberRight
          obtain ⟨leftTail, leftEq, _, _, _⟩ := (mem_enumerate relation need _ _ _ (valid.append leftAllowed)).mp memberLeft
          obtain ⟨rightTail, rightEq, _, _, _⟩ := (mem_enumerate relation need _ _ _ (valid.append rightAllowed)).mp memberRight
          rw [leftEq, rightEq, List.append_assoc, List.append_assoc]
          exact lex_append_front front (List.Lex.rel less)
        · simp [rightAllowed] at memberRight
      · simp [leftAllowed] at memberLeft

theorem select_bound (options : List (List Nat)) (best : Best) : (select options best).bound ≤ best.bound := by
  induction options generalizing best with
  | nil => exact Int.le_refl _
  | cons chosen rest ih => exact Int.le_trans (ih _) (improve_bound chosen (chosen.sum : Int) best)

theorem select_equal_bound (options : List (List Nat)) (best : Best)
    (same : (select options best).bound = best.bound) : select options best = best := by
  induction options generalizing best with
  | nil => rfl
  | cons chosen rest ih =>
    let updated := improve chosen (chosen.sum : Int) best
    have reduced : updated.bound ≤ best.bound := improve_bound chosen (chosen.sum : Int) best
    have finalBound := select_bound rest updated
    change (select rest updated).bound = best.bound at same
    have equal : updated.bound = best.bound := by omega
    have unchanged := improve_equal_bound chosen (chosen.sum : Int) best equal
    exact (ih updated (by omega)).trans unchanged

theorem select_sound (relation : Nat → Nat → Bool) (need : Nat) (values : List Nat)
    (options : List (List Nat)) (best : Best) (valid : ∀ chosen ∈ options, Completion relation need values chosen) :
    Sound relation need [] values 0 best (select options best) := by
  induction options generalizing best with
  | nil => exact Sound.refl _ _ _ _ _ _
  | cons chosen rest ih =>
    let updated := improve chosen (chosen.sum : Int) best
    have first : Sound relation need [] values 0 best updated := by
      by_cases improves : (chosen.sum : Int) < best.bound
      · exact Or.inr ⟨chosen, valid chosen (by simp), by simp [updated, improve, improves],
          by simp [updated, improve, improves], by simpa only [updated, improve, if_pos improves] using improves⟩
      · exact Or.inl (by simp only [updated, improve, if_neg improves])
    exact first.trans (ih updated (fun candidate member => valid candidate (List.mem_cons_of_mem chosen member)))

theorem improve_le_total (chosen : List Nat) (best : Best) : (improve chosen (chosen.sum : Int) best).bound ≤ (chosen.sum : Int) := by
  unfold improve
  split
  · exact Int.le_refl _
  · omega

theorem select_le_member (options : List (List Nat)) (best : Best) (chosen : List Nat) (member : chosen ∈ options) :
    (select options best).bound ≤ (chosen.sum : Int) := by
  induction options generalizing best with
  | nil => simp at member
  | cons head rest ih =>
    rcases List.mem_cons.mp member with same | member
    · subst chosen
      exact Int.le_trans (select_bound rest _) (improve_le_total head best)
    · exact ih _ member

theorem select_first (options : List (List Nat)) (best : Best) (ordered : options.Pairwise (List.Lex (· < ·)))
    (chosen : List Nat) (member : chosen ∈ options) (total : (chosen.sum : Int) = (select options best).bound)
    (improves : (select options best).bound < best.bound) : Precedes (select options best).chosen chosen := by
  induction options generalizing best with
  | nil => simp at member
  | cons head rest ih =>
    let updated := improve head (head.sum : Int) best
    have finalBound := select_bound rest updated
    have sorted := List.pairwise_cons.mp ordered
    change (select rest updated).bound < best.bound at improves
    change (chosen.sum : Int) = (select rest updated).bound at total
    change Precedes (select rest updated).chosen chosen
    rcases List.mem_cons.mp member with same | member
    · subst chosen
      have headBetter : (head.sum : Int) < best.bound := by omega
      have updatedValue : updated = ⟨(head.sum : Int), head⟩ := by simp only [updated, improve, if_pos headBetter]
      have updatedBound : updated.bound = (head.sum : Int) := congrArg Best.bound updatedValue
      have equal : (select rest updated).bound = updated.bound := by omega
      have same := select_equal_bound rest updated equal
      rw [same, updatedValue]
      exact Or.inl rfl
    · by_cases improvedAgain : (select rest updated).bound < updated.bound
      · exact ih updated sorted.2 member total improvedAgain
      · have equal : (select rest updated).bound = updated.bound := by omega
        have same := select_equal_bound rest updated equal
        have updatedImproves : updated.bound < best.bound := by omega
        by_cases headBetter : (head.sum : Int) < best.bound
        · rw [same]
          simp only [updated, improve, if_pos headBetter]
          exact Or.inr (sorted.1 chosen member)
        · simp only [updated, improve, if_neg headBetter] at updatedImproves
          omega

theorem exhaustive_optimal (relation : Nat → Nat → Bool) (need : Nat) (values : List Nat) (best : Best)
    (ordered : values.Pairwise (· < ·)) : Optimal relation need [] values 0 best (exhaustive relation need [] values best) := by
  refine ⟨select_sound relation need values _ best (fun chosen member => (mem_enumerate_root _ _ _ _ ordered).mp member), ?_, ?_⟩
  · intro chosen completion
    simpa only [Int.zero_add] using select_le_member (enumerate relation need [] values) best chosen
      ((mem_enumerate_root _ _ _ _ ordered).mpr completion)
  · intro chosen completion total improves
    exact select_first _ best (enumerate_increasing relation need [] values ⟨by simp, by simp⟩ ordered) chosen
      ((mem_enumerate_root _ _ _ _ ordered).mpr completion) (by simpa only [Int.zero_add] using total) improves

theorem exhaustive_eq_search (relation : Nat → Nat → Bool) (need : Nat) (values : List Nat) (best : Best)
    (ordered : values.Pairwise (· < ·)) :
    exhaustive relation need [] values best = search relation need [] values 0 best :=
  (exhaustive_optimal relation need values best ordered).unique (search_optimal relation need [] values 0 best ordered)

end ProjectEulerSolutions.CliqueEnumeration
