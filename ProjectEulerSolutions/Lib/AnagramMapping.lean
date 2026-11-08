import ProjectEulerSolutions.Lib.AnagramAssignment
import Mathlib.Data.List.Nodup

/-! Incremental bijection checks and canonical equality patterns accept the same assignments. -/
namespace ProjectEulerSolutions.AnagramMapping
open AnagramSquares
set_option Elab.async false

variable {α β : Type*} [DecidableEq α] [DecidableEq β]

def check (letters : List α) (digits : List β) : List α → List β → Option (List α × List β)
  | [], [] => some (letters, digits)
  | letter :: rest, digit :: tail =>
      if letters.idxOf letter = digits.idxOf digit then
        check (extend letters letter) (extend digits digit) rest tail
      else none
  | _, _ => none

theorem check_eq_labels (letters : List α) (digits : List β) (word : List α) (value : List β) :
    check letters digits word value =
      if labels letters word = labels digits value then some (alphabet letters word, alphabet digits value) else none := by
  induction word generalizing letters digits value with
  | nil => cases value <;> simp [check, labels, alphabet]
  | cons letter rest ih =>
      cases value with
      | nil => simp [check, labels]
      | cons digit tail =>
          by_cases same : letters.idxOf letter = digits.idxOf digit <;>
            simp [check, labels, alphabet, same, ih]

theorem check_succeeds_iff (word : List α) (value : List β) :
    (check [] [] word value).isSome = true ↔ labels [] word = labels [] value := by
  rw [check_eq_labels]
  split <;> simp_all

theorem check_iff_substitution (fallback : β) (word : List α) (value : List β) :
    (check [] [] word value).isSome = true ↔
      ∃ f : α → β, word.map f = value ∧ Set.InjOn f {letter | letter ∈ word} :=
  (check_succeeds_iff word value).trans (labels_eq_iff_substitution fallback word value)

/-- An existing letter must keep its digit; a new letter must receive an unused digit. -/
theorem index_match (letters : List α) (digits : List β) (letter : α) (digit : β)
    (sameLength : letters.length = digits.length) (uniqueDigits : digits.Nodup) :
    letters.idxOf letter = digits.idxOf digit ↔
      if letter ∈ letters then digits[letters.idxOf letter]? = some digit else digit ∉ digits := by
  by_cases member : letter ∈ letters
  · simp only [if_pos member]
    have within : letters.idxOf letter < digits.length := by
      rw [← sameLength]
      exact List.idxOf_lt_length_of_mem member
    constructor
    · intro same
      have digitWithin : digits.idxOf digit < digits.length := by omega
      rw [same, List.getElem?_eq_getElem digitWithin, List.getElem_idxOf digitWithin]
    · intro found
      rw [List.getElem?_eq_getElem within] at found
      have equal : digits[letters.idxOf letter] = digit := Option.some.inj found
      have indexOf := List.get_idxOf uniqueDigits ⟨letters.idxOf letter, within⟩
      change digits.idxOf digits[letters.idxOf letter] = letters.idxOf letter at indexOf
      rw [equal] at indexOf
      exact indexOf.symm
  · simp only [if_neg member, List.idxOf_of_notMem member, sameLength]
    exact eq_comm.trans List.idxOf_eq_length_iff

theorem extend_nodup (known : List α) (letter : α) (unique : known.Nodup) :
    (extend known letter).Nodup := by
  by_cases member : letter ∈ known
  · simpa [extend, member] using unique
  · simp [extend, member, List.nodup_append, unique]
    intro other present same
    exact member (same ▸ present)

theorem extend_lengths (letters : List α) (digits : List β) (letter : α) (digit : β)
    (sameLength : letters.length = digits.length) (sameIndex : letters.idxOf letter = digits.idxOf digit) :
    (extend letters letter).length = (extend digits digit).length := by
  have membership : letter ∈ letters ↔ digit ∈ digits := by
    rw [← List.idxOf_lt_length_iff, ← List.idxOf_lt_length_iff, sameIndex, sameLength]
  by_cases member : letter ∈ letters
  · simp [extend, member, membership.mp member, sameLength]
  · simp [extend, member, membership.not.mp member, sameLength]

theorem check_preserves (letters : List α) (digits : List β) (word : List α) (value : List β)
    (sameLength : letters.length = digits.length) (uniqueLetters : letters.Nodup) (uniqueDigits : digits.Nodup)
    (result : List α × List β) (success : check letters digits word value = some result) :
    result.1.length = result.2.length ∧ result.1.Nodup ∧ result.2.Nodup := by
  induction word generalizing letters digits value with
  | nil =>
      cases value with
      | nil => cases success; exact ⟨sameLength, uniqueLetters, uniqueDigits⟩
      | cons _ _ => simp [check] at success
  | cons letter rest ih =>
      cases value with
      | nil => simp [check] at success
      | cons digit tail =>
          simp only [check] at success
          split at success
          · rename_i sameIndex
            exact ih _ _ _ (extend_lengths letters digits letter digit sameLength sameIndex)
              (extend_nodup letters letter uniqueLetters) (extend_nodup digits digit uniqueDigits) success
          · cases success

/-- The completed first-occurrence table gives the same digit as the mathematical substitution. -/
theorem assignment_lookup (fallback : β) (word : List α) (value : List β) (letter : α)
    (matched : labels [] word = labels [] value) (member : letter ∈ word) :
    (alphabet [] value)[(alphabet [] word).idxOf letter]? = some (substitution fallback word value letter) := by
  have sameLength : word.length = value.length := by simpa using congrArg List.length matched
  have inWord : word.idxOf letter < word.length := List.idxOf_lt_length_of_mem member
  have inValue : word.idxOf letter < value.length := by omega
  have atLetter : word[word.idxOf letter] = letter := List.getElem_idxOf inWord
  have indices : (alphabet [] word).idxOf letter = (alphabet [] value).idxOf value[word.idxOf letter] := by
    rw [labels_eq_map, labels_eq_map] at matched
    have atPosition := congrArg (fun keys : List Nat => keys[word.idxOf letter]?) matched
    simpa only [List.getElem?_map, List.getElem?_eq_getElem inWord, List.getElem?_eq_getElem inValue,
      Option.map_some, Option.some.injEq, atLetter] using atPosition
  rw [indices, List.getElem?_idxOf (mem_alphabet [] value _ (List.getElem_mem inValue))]
  simp [substitution, List.getElem?_eq_getElem inValue]

end ProjectEulerSolutions.AnagramMapping
