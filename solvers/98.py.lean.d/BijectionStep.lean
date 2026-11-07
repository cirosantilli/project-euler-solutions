import solvers.«98.py.lean.d».Others
import data.«project-euler-statements».data.py.project_euler_statements.«p98.py.lean.d».Mapping

namespace solvers.«98».bijection
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.AnagramSquares ProjectEulerSolutions.AnagramMapping
open pattern_signature (program)
open data.«project-euler-statements».data.py.project_euler_statements.p98.mapping
  (character value table lookup_table insert_table)
set_option Elab.async false

def work (base : Mutable.Heap) (letters digits : List Char) (garbage : Mutable.Heap) : Mutable.Heap :=
  base ++ (.dict (table letters digits) :: .dict (table digits letters) :: garbage)

@[simp] theorem found_forward (base garbage : Mutable.Heap) (letters digits : List Char) :
    (work base letters digits garbage)[base.length]? = some (.dict (table letters digits)) := by simp [work]

@[simp] theorem found_reverse (base garbage : Mutable.Heap) (letters digits : List Char) :
    (work base letters digits garbage)[base.length + 1]? = some (.dict (table digits letters)) := by simp [work]

theorem insert_lookup (entries : List (Constant × Mutable.Value)) (key : Constant) (stored : Mutable.Value)
    (found : ScalarDict.lookup entries key = some stored) : ScalarDict.insert entries key stored = entries := by
  induction entries with
  | nil => simp [ScalarDict.lookup] at found
  | cons entry rest ih =>
      obtain ⟨first, previous⟩ := entry
      simp only [ScalarDict.lookup, ScalarDict.insert] at found ⊢
      split at found <;> simp_all

theorem lookup_matched (letters digits : List Char) (letter digit : Char)
    (sameLength : letters.length = digits.length) (uniqueDigits : digits.Nodup)
    (matched : letters.idxOf letter = digits.idxOf digit) :
    ScalarDict.lookup (table letters digits) (character letter) = none ∨
      ScalarDict.lookup (table letters digits) (character letter) = some (value digit) := by
  have checked := (index_match letters digits letter digit sameLength uniqueDigits).mp matched
  rw [lookup_table letters digits letter sameLength]
  by_cases member : letter ∈ letters
  · right
    have selected : digits[letters.idxOf letter]? = some digit := by simpa only [if_pos member] using checked
    simp [selected]
  · left
    simp [List.idxOf_of_notMem member, sameLength]

theorem insert_matched (letters digits : List Char) (letter digit : Char)
    (sameLength : letters.length = digits.length) (uniqueDigits : digits.Nodup)
    (matched : letters.idxOf letter = digits.idxOf digit) :
    ScalarDict.insert (table letters digits) (character letter) (value digit) =
      table (extend letters letter) (extend digits digit) := by
  have checked := (index_match letters digits letter digit sameLength uniqueDigits).mp matched
  by_cases member : letter ∈ letters
  · have selected : digits[letters.idxOf letter]? = some digit := by simpa only [if_pos member] using checked
    have digitMember := List.mem_of_getElem? selected
    simp only [extend, if_pos member, if_pos digitMember]
    apply insert_lookup
    rw [lookup_table letters digits letter sameLength, selected]
    rfl
  · have absent : digit ∉ digits := by simpa only [if_neg member] using checked
    simpa only [extend, if_neg member, if_neg absent] using insert_table letters digits letter digit sameLength member

abbrev collision (dictionary key expected : String) : Expr :=
  .boolOp .and [.membership (.name key) (.name dictionary) false,
    .compare (.subscript (.name dictionary) (.name key)) [(.notEq, .name expected)]]

abbrev reject : List Stmt := [.assign [.name "ok"] (.constant (.bool false)), .breakLoop]
abbrev record : List Stmt :=
  [.assign [.subscript (.name "char_to_digit") (.name "ch")] (.name "d"),
   .assign [.subscript (.name "digit_to_char") (.name "d")] (.name "ch")]
abbrev body : List Stmt :=
  [.ifThen (collision "char_to_digit" "ch" "d") reject [],
   .ifThen (collision "digit_to_char" "d" "ch") reject []] ++ record
abbrev candidateBody : List Stmt := match (candidates.wordBody[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []

theorem source : candidateBody[3]? =
    some (.forZip (.tuple [.name "ch", .name "d"]) [.name "w1", .name "sq_str"] body []) := rfl

structure Locals (env : HeapCalls.Env) (base : Mutable.Heap) (letter digit : Char) : Prop where
  forward : env "char_to_digit" = some (.ref base.length)
  reverse : env "digit_to_char" = some (.ref (base.length + 1))
  getLetter : env "ch" = some (value letter)
  getDigit : env "d" = some (value digit)

theorem collision_false (heap : Mutable.Heap) (address : Nat) (entries : List (Constant × Mutable.Value))
    (letter digit : Char) (env : HeapCalls.Env) (dictionary key expected : String)
    (found : heap[address]? = some (.dict entries))
    (getDictionary : env dictionary = some (.ref address)) (getKey : env key = some (value letter))
    (getExpected : env expected = some (value digit))
    (correct : ScalarDict.lookup entries (character letter) = none ∨
      ScalarDict.lookup entries (character letter) = some (value digit)) :
    Evaluates program env (collision dictionary key expected) heap (.scalar (.bool false)) heap := by
  have member : Evaluates program env (.membership (.name key) (.name dictionary) false) heap
      (.scalar (.bool (ScalarDict.lookup entries (character letter)).isSome)) heap := by
    apply Evaluates.membership (.cons (Evaluates.lookup getKey) (.cons (Evaluates.lookup getDictionary) .nil))
    simp [HeapCalls.membership, found]
  rcases correct with absent | present
  · rw [absent] at member
    exact Evaluates.boolean_stop member rfl rfl
  · rw [present] at member
    apply Evaluates.boolean_next member rfl rfl
    apply Evaluates.boolean_last (flag := false) (truth := rfl)
    apply Evaluates.compare (av := value digit) (bv := value digit) (middle := heap) (afterRight := heap)
    · apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getDictionary) (Evaluates.lookup getKey)
      simp [Mutable.getItem, found, present]
    · exact Evaluates.lookup getExpected
    · simp [HeapCalls.compare, scalarCompare]

/-- Both source stores retain the first occurrence order and extend the inverse tables together. -/
theorem record_result (base garbage : Mutable.Heap) (letters digits : List Char) (letter digit : Char)
    (env : HeapCalls.Env) (locals : Locals env base letter digit)
    (sameLength : letters.length = digits.length) (uniqueLetters : letters.Nodup) (uniqueDigits : digits.Nodup)
    (matched : letters.idxOf letter = digits.idxOf digit) :
    BlockContinues program env record (work base letters digits garbage) env
      (work base (extend letters letter) (extend digits digit) garbage) := by
  let middle := base ++ (.dict (table (extend letters letter) (extend digits digit)) ::
    .dict (table digits letters) :: garbage)
  have first : BlockContinues program env (record.take 1) (work base letters digits garbage) env middle := by
    apply BlockContinues.setItem (by intros; simp) (Evaluates.lookup locals.getDigit)
      (.cons (Evaluates.lookup locals.forward) (.cons (Evaluates.lookup locals.getLetter) .nil))
    simp [Mutable.setItem, insert_matched letters digits letter digit sameLength uniqueDigits matched,
      work, middle, List.set_append_right, Except.map]
  apply first.append
  apply BlockContinues.setItem (by intros; simp) (Evaluates.lookup locals.getLetter)
    (.cons (Evaluates.lookup locals.reverse) (.cons (Evaluates.lookup locals.getDigit) .nil))
  simp [Mutable.setItem, insert_matched digits letters digit letter sameLength.symm uniqueLetters matched.symm,
    work, middle, List.set_append_right, Except.map]

/-- Matching patterns make both short-circuit collision guards false in the actual Python body. -/
theorem step_success (base garbage : Mutable.Heap) (letters digits : List Char) (letter digit : Char)
    (env : HeapCalls.Env) (locals : Locals env base letter digit)
    (sameLength : letters.length = digits.length) (uniqueLetters : letters.Nodup) (uniqueDigits : digits.Nodup)
    (matched : letters.idxOf letter = digits.idxOf digit) :
    BlockContinues program env body (work base letters digits garbage) env
      (work base (extend letters letter) (extend digits digit) garbage) := by
  have forward := collision_false (work base letters digits garbage) base.length (table letters digits)
    letter digit env "char_to_digit" "ch" "d" (found_forward _ _ _ _) locals.forward locals.getLetter locals.getDigit
    (lookup_matched letters digits letter digit sameLength uniqueDigits matched)
  have reverse := collision_false (work base letters digits garbage) (base.length + 1) (table digits letters)
    digit letter env "digit_to_char" "d" "ch" (found_reverse _ _ _ _) locals.reverse locals.getDigit locals.getLetter
    (lookup_matched digits letters digit letter sameLength.symm uniqueLetters matched.symm)
  exact (BlockContinues.branch (flag := false) forward rfl BlockContinues.nil).append
    ((BlockContinues.branch (flag := false) reverse rfl BlockContinues.nil).append
      (record_result base garbage letters digits letter digit env locals sameLength uniqueLetters uniqueDigits matched))

end solvers.«98».bijection
