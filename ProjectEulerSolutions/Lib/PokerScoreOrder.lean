import ProjectEulerSolutions.Lib.PokerTiebreaks

namespace ProjectEulerSolutions.PokerScoreOrder
open FormalSW.PyAst
open ProjectEulerSolutions.PokerTiebreaks
set_option Elab.async false

structure Score where
  tag : Int
  key : List Int

def weights (tag : Int) : List Nat :=
  if tag = 7 then [4, 1]
  else if tag = 6 then [3, 2]
  else if tag = 3 then [3, 1, 1]
  else if tag = 2 then [2, 2, 1]
  else if tag = 1 then [2, 1, 1, 1]
  else if tag = 4 ∨ tag = 8 then [1]
  else [1, 1, 1, 1, 1]

structure Valid (score : Score) : Prop where
  nonnegative : 0 ≤ score.tag
  upper : score.tag ≤ 8
  keyLength : score.key.length = (weights score.tag).length
  straightFlushHigh : score.tag = 8 → score.key.headD 0 ≤ 14

abbrev Royal (score : Score) : Prop := score.tag = 8 ∧ score.key = [14]

/-- The reference expands multiplicities and gives the ace-high straight flush its separate category. -/
def normalize (score : Score) : Score :=
  if Royal score then ⟨9, []⟩ else ⟨score.tag, expand (weights score.tag) score.key⟩

def comparison (left right : Score) : Bool :=
  if left.tag = right.tag then referenceGreater left.key right.key else decide (right.tag < left.tag)

theorem weights_positive (tag : Int) (copies : Nat) (member : copies ∈ weights tag) : 0 < copies := by
  unfold weights at member
  split at member
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    omega
  · split at member
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      omega
    · split at member
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at member
        omega
      · split at member
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at member
          omega
        · split at member
          · simp only [List.mem_cons, List.not_mem_nil, or_false] at member
            omega
          · split at member
            · simp only [List.mem_cons, List.not_mem_nil, or_false] at member
              omega
            · simp only [List.mem_cons, List.not_mem_nil, or_false] at member
              omega

theorem straight_flush_key (score : Score) (valid : Valid score) (tag : score.tag = 8) :
    ∃ high, score.key = [high] ∧ high ≤ 14 := by
  have length : score.key.length = 1 := by simpa [weights, tag] using valid.keyLength
  cases shape : score.key with
  | nil => simp [shape] at length
  | cons high rest =>
      have empty : rest = [] := by simpa [shape] using length
      subst rest
      exact ⟨high, rfl, by simpa [shape] using valid.straightFlushHigh tag⟩

/-- The bounded straight-flush high card makes the separate royal category strictly greatest. -/
theorem below_royal (score : Score) (valid : Valid score) (ordinary : ¬ Royal score) :
    HeapSort.lexInt (score.tag :: score.key) [8, 14] = true ∧
      HeapSort.lexInt [8, 14] (score.tag :: score.key) = false := by
  by_cases top : score.tag = 8
  · obtain ⟨high, key, upper⟩ := straight_flush_key score valid top
    have different : high ≠ 14 := by
      intro same
      exact ordinary ⟨top, by simpa only [same] using key⟩
    have less : high < 14 := by omega
    have notGreater : ¬ (14 : Int) < high := by omega
    simp [HeapSort.lexInt, top, key, different, Ne.symm different, less, notGreater]
  · have less : score.tag < 8 := by have bound := valid.upper; omega
    have notGreater : ¬ (8 : Int) < score.tag := by omega
    simp [HeapSort.lexInt, top, Ne.symm top, less, notGreater]

theorem expanded_order (left right : Score) (leftValid : Valid left) (rightValid : Valid right)
    (leftOrdinary : ¬ Royal left) (rightOrdinary : ¬ Royal right) :
    comparison (normalize left) (normalize right) = HeapSort.lexInt (right.tag :: right.key) (left.tag :: left.key) := by
  change (if (normalize left).tag = (normalize right).tag then
    referenceGreater (normalize left).key (normalize right).key else
    decide ((normalize right).tag < (normalize left).tag)) = _
  simp only [normalize, if_neg leftOrdinary, if_neg rightOrdinary]
  by_cases same : left.tag = right.tag
  · rw [if_pos same]
    have sameWeights : weights right.tag = weights left.tag := congrArg weights same.symm
    rw [sameWeights]
    rw [expanded_referenceGreater (weights left.tag) left.key right.key leftValid.keyLength
      (by simpa only [sameWeights] using rightValid.keyLength) (weights_positive left.tag)]
    simp only [HeapSort.lexInt, if_pos same.symm]
  · rw [if_neg same]
    simp only [HeapSort.lexInt, if_neg (Ne.symm same)]

/-- Every supported poker category, including the reference's royal-flush adjustment, preserves the winner. -/
theorem normalized_comparison (left right : Score) (leftValid : Valid left) (rightValid : Valid right) :
    comparison (normalize left) (normalize right) = HeapSort.lexInt (right.tag :: right.key) (left.tag :: left.key) := by
  by_cases leftRoyal : Royal left
  · by_cases rightRoyal : Royal right
    · have leftTag : left.tag = 8 := leftRoyal.1
      have leftKey : left.key = [14] := leftRoyal.2
      have rightTag : right.tag = 8 := rightRoyal.1
      have rightKey : right.key = [14] := rightRoyal.2
      simp only [normalize, if_pos leftRoyal, if_pos rightRoyal]
      change false = HeapSort.lexInt (right.tag :: right.key) (left.tag :: left.key)
      rw [leftTag, leftKey, rightTag, rightKey]
      rfl
    · have rightOrder := (below_royal right rightValid rightRoyal).1
      have rootTrue : HeapSort.lexInt (right.tag :: right.key) (left.tag :: left.key) = true := by
        simpa only [leftRoyal.1, leftRoyal.2] using rightOrder
      have different : (9 : Int) ≠ right.tag := by have bound := rightValid.upper; omega
      have less : right.tag < 9 := by have bound := rightValid.upper; omega
      simp only [normalize, if_pos leftRoyal, if_neg rightRoyal]
      simp [comparison, different, less, rootTrue]
  · by_cases rightRoyal : Royal right
    · have leftOrder := (below_royal left leftValid leftRoyal).2
      have rootFalse : HeapSort.lexInt (right.tag :: right.key) (left.tag :: left.key) = false := by
        simpa only [rightRoyal.1, rightRoyal.2] using leftOrder
      have different : left.tag ≠ (9 : Int) := by have bound := leftValid.upper; omega
      have notLess : ¬ (9 : Int) < left.tag := by have bound := leftValid.upper; omega
      simp only [normalize, if_neg leftRoyal, if_pos rightRoyal]
      simp [comparison, different, notLess, rootFalse]
    · exact expanded_order left right leftValid rightValid leftRoyal rightRoyal

end ProjectEulerSolutions.PokerScoreOrder
