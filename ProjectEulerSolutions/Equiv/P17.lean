import ProjectEulerSolutions.P17
import ProjectEulerStatements.Lib.NumberWords

/-! Equivalence on the number-name domain accepted by the Python solver. -/
namespace ProjectEulerSolutions.P17
open ProjectEulerStatements.P17

private theorem ones_lookup (n : Nat) (bound : n < 10) : ones[n]! = smallWordLen n := by
  interval_cases n <;> rfl

private theorem teens_lookup (n : Nat) (lower : 10 ≤ n) (upper : n < 20) :
    teens[n - 10]! = smallWordLen n := by
  interval_cases n <;> rfl

private theorem tens_lookup (n : Nat) (lower : 2 ≤ n) (upper : n < 10) :
    tens[n - 2]! = tensWordLen n := by
  interval_cases n <;> rfl

theorem letters1To99_eq (n : Nat) (bound : n < 100) : letters1To99 n = wordLen n := by
  unfold letters1To99
  by_cases onesCase : n < 10
  · rw [if_pos onesCase, ones_lookup n onesCase, wordLen_below_twenty n (by omega)]
  · rw [if_neg onesCase]
    by_cases teensCase : n < 20
    · rw [if_pos teensCase, teens_lookup n (by omega) teensCase, wordLen_below_twenty n teensCase]
    · rw [if_neg teensCase]
      dsimp only
      rw [tens_lookup (n / 10) (by omega) (by omega),
        ones_lookup (n % 10) (Nat.mod_lt _ (by decide)), wordLen_below_hundred n bound,
        if_neg teensCase]

theorem lettersInNumber_eq (n : Nat) (bound : n ≤ 1000) : lettersInNumber n = wordLen n := by
  unfold lettersInNumber
  by_cases thousand : n = 1000
  · subst n
    simp
  · simp only [beq_iff_eq, thousand, if_false]
    by_cases small : n < 100
    · rw [if_pos small, letters1To99_eq n small]
    · rw [if_neg small]
      rw [ones_lookup (n / 100) (by omega), wordLen_hundreds n (by omega) (by omega)]
      by_cases round : n % 100 = 0
      · simp [round]
      · simp [round, letters1To99_eq (n % 100) (Nat.mod_lt _ (by decide)), Nat.add_assoc]
        omega

/-- The bound comes from the source's explicit number-name assertions. -/
theorem equiv (n : Nat) (bound : n ≤ 1000) : ProjectEulerStatements.P17.naive n = solve n := by
  unfold ProjectEulerStatements.P17.naive solve
  have values : (List.range (n + 1)).map wordLen = (List.range (n + 1)).map lettersInNumber := by
    apply List.map_congr_left
    intro i member
    exact (lettersInNumber_eq i (by have := List.mem_range.mp member; omega)).symm
  simpa only [List.foldl_map] using congrArg (fun xs : List Nat => xs.foldl (· + ·) 0) values

end ProjectEulerSolutions.P17
