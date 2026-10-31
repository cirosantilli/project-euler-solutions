import ProjectEulerSolutions.Lib.RationalOperations

/-! The reference enumerates exact rational pairs without needing gcd normalization. -/
namespace ProjectEulerSolutions.RationalPair
set_option Elab.async false

def combine (left right : List Pair) : List Pair :=
  left.flatMap fun a => right.flatMap fun b => reference a b

theorem combine_positive (left right : List Pair)
    (leftPositive : ∀ a ∈ left, 0 < a.2) (rightPositive : ∀ b ∈ right, 0 < b.2)
    (pair : Pair) (member : pair ∈ combine left right) : 0 < pair.2 := by
  obtain ⟨a, ha, member⟩ := List.mem_flatMap.mp member
  obtain ⟨b, hb, member⟩ := List.mem_flatMap.mp member
  exact reference_positive a b pair (leftPositive a ha) (rightPositive b hb) member

theorem combine_values (left right : List Pair)
    (leftPositive : ∀ a ∈ left, 0 < a.2) (rightPositive : ∀ b ∈ right, 0 < b.2) :
    (combine left right).map value =
      (left.map value).flatMap fun a => (right.map value).flatMap fun b => ArithmeticExpressions.operations a b := by
  simp only [combine, List.map_flatMap, List.flatMap_map]
  apply List.flatMap_congr
  intro a ha
  apply List.flatMap_congr
  intro b hb
  exact reference_values a b (leftPositive a ha) (rightPositive b hb)

/-- Ordered expression results, retaining duplicates and positive unreduced denominators. -/
def enumeratePairs (xs : List Int) : List Pair :=
  match xs with
  | [] => []
  | [a] => [(a, 1)]
  | a :: b :: rest =>
      (List.range' 1 (rest.length + 1)).attach.flatMap fun (split : {i // i ∈ List.range' 1 (rest.length+1)}) =>
        have lower : 0 < split.val := by
          obtain ⟨i, bound, equal⟩ := List.mem_range'.mp split.property
          omega
        have upper : split.val < (a :: b :: rest).length := by
          obtain ⟨i, bound, equal⟩ := List.mem_range'.mp split.property
          simp only [List.length_cons]
          omega
        combine (enumeratePairs ((a :: b :: rest).take split.val)) (enumeratePairs ((a :: b :: rest).drop split.val))
termination_by xs.length
decreasing_by
  all_goals simp_wf
  all_goals simp only [List.length_cons] at *
  all_goals omega

private theorem flatMap_attach_values {α β : Type} (xs : List α) (f : α → List β) :
    xs.attach.flatMap (fun item => f item.val) = xs.flatMap f := by
  have same : xs.attach.map Subtype.val = xs := by simp
  simpa only [List.flatMap_map] using congrArg (fun items => items.flatMap f) same

theorem enumeratePairs_step (a b : Int) (rest : List Int) :
    enumeratePairs (a :: b :: rest) = (List.range' 1 (rest.length+1)).flatMap fun split =>
      combine (enumeratePairs ((a :: b :: rest).take split)) (enumeratePairs ((a :: b :: rest).drop split)) := by
  rw [enumeratePairs]
  exact flatMap_attach_values (List.range' 1 (rest.length+1)) (fun split =>
    combine (enumeratePairs ((a :: b :: rest).take split)) (enumeratePairs ((a :: b :: rest).drop split)))

private theorem enumerate_step (a b : Rat) (rest : List Rat) :
    ArithmeticExpressions.enumerate (a :: b :: rest) = (List.range' 1 (rest.length+1)).flatMap fun split =>
      (ArithmeticExpressions.enumerate ((a :: b :: rest).take split)).flatMap fun left =>
        (ArithmeticExpressions.enumerate ((a :: b :: rest).drop split)).flatMap fun right =>
          ArithmeticExpressions.operations left right := by
  rw [ArithmeticExpressions.enumerate]
  exact flatMap_attach_values (List.range' 1 (rest.length+1)) (fun split =>
    (ArithmeticExpressions.enumerate ((a :: b :: rest).take split)).flatMap fun left =>
      (ArithmeticExpressions.enumerate ((a :: b :: rest).drop split)).flatMap fun right =>
        ArithmeticExpressions.operations left right)

/-- Every recursive result is a valid rational, and the complete value list matches tree enumeration. -/
theorem enumeratePairs_spec (xs : List Int) :
    (∀ pair ∈ enumeratePairs xs, 0 < pair.2) ∧
    (enumeratePairs xs).map value = ArithmeticExpressions.enumerate (xs.map (fun (n : Int) => (n : Rat))) := by
  refine (measure List.length).wf.induction
    (C := fun xs => (∀ pair ∈ enumeratePairs xs, 0 < pair.2) ∧
      (enumeratePairs xs).map value = ArithmeticExpressions.enumerate (xs.map (fun (n : Int) => (n : Rat)))) xs ?_
  intro xs ih
  cases xs with
  | nil => simp [enumeratePairs, ArithmeticExpressions.enumerate]
  | cons a tail =>
      cases tail with
      | nil => simp [enumeratePairs, ArithmeticExpressions.enumerate, value]
      | cons b rest =>
          have children (split : {i // i ∈ List.range' 1 (rest.length+1)}) :
              (∀ pair ∈ enumeratePairs ((a :: b :: rest).take split.val), 0 < pair.2) ∧
              (enumeratePairs ((a :: b :: rest).take split.val)).map value =
                ArithmeticExpressions.enumerate (((a :: b :: rest).take split.val).map (fun (n : Int) => (n : Rat))) ∧
              (∀ pair ∈ enumeratePairs ((a :: b :: rest).drop split.val), 0 < pair.2) ∧
              (enumeratePairs ((a :: b :: rest).drop split.val)).map value =
                ArithmeticExpressions.enumerate (((a :: b :: rest).drop split.val).map (fun (n : Int) => (n : Rat))) := by
            obtain ⟨i, bound, equal⟩ := List.mem_range'.mp split.property
            have left := ih ((a :: b :: rest).take split.val) (by
              change List.length _ < List.length _
              simp only [List.length_take, List.length_cons]
              omega)
            have right := ih ((a :: b :: rest).drop split.val) (by
              change List.length _ < List.length _
              simp only [List.length_drop, List.length_cons]
              omega)
            exact ⟨left.1, left.2, right.1, right.2⟩
          constructor
          · intro pair member
            rw [enumeratePairs] at member
            obtain ⟨split, _, member⟩ := List.mem_flatMap.mp member
            exact combine_positive _ _ (children split).1 (children split).2.2.1 pair member
          · rw [enumeratePairs_step]
            simp only [List.map_flatMap, List.map_cons, enumerate_step, List.length_map]
            apply List.flatMap_congr
            intro split member
            rw [combine_values _ _ (children ⟨split, member⟩).1 (children ⟨split, member⟩).2.2.1,
              (children ⟨split, member⟩).2.1, (children ⟨split, member⟩).2.2.2]
            simp only [List.map_take, List.map_drop, List.map_cons]

end ProjectEulerSolutions.RationalPair
