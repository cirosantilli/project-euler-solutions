import solvers.«89.py.lean.d».ParserState
import ProjectEulerSolutions.Lib.RomanParsing

namespace solvers.«89».roman_to_int
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false
set_option maxHeartbeats 1600000

abbrev suffix (text : String) (index : Nat) : Nat :=
  RomanNumerals.parse ((text.toList.map String.singleton).drop index)

abbrev loopCode : List Stmt := [code[3]'(by decide), code[4]'(by decide)]
abbrev additiveCode : List Stmt := body.drop 2 ++ [.loopResume (code[3]'(by decide)), code[4]'(by decide)]

/-- The normal path consumes one symbol and resumes the original while statement. -/
theorem add_one (text : String) (index total value : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (ready : Locals env globals text index total)
    (getValue : env "v" = some (number value)) (answer : Mutable.Value)
    (rest : BlockReturns program
      (setLocal (setLocal env "total" (number (total + value))) "i" (number (index + 1)))
      loopCode heap answer heap globals) :
    BlockReturns program env additiveCode heap answer heap globals := by
  apply BlockReturns.augAssign (middle := heap) (afterOperation := heap) ready.total (Evaluates.lookup getValue)
    (value := number (total + value))
  · simp [inplace, Mutable.binary, scalarBinary, number]
  apply BlockReturns.augAssign (middle := heap) (afterOperation := heap) (left := number index) (value := number (index + 1))
  · simpa [setLocal] using ready.position
  · exact Evaluates.constant
  · simp [inplace, Mutable.binary, scalarBinary, number]
  exact BlockReturns.loop_resume rest

/-- Every iteration decreases the unconsumed suffix; the input and caller heap are unchanged. -/
theorem loop (text : String) (valid : RomanText text) (index total : Nat)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (table : SymbolTable heap globals)
    (env : HeapCalls.Env) (ready : Locals env globals text index total) :
    BlockReturns program env loopCode heap (number (total + suffix text index)) heap globals := by
  by_cases inside : index < text.toList.length
  · apply BlockReturns.while_loop (flag := true) (active text index total heap globals env ready)
      (by simp only [inside]; rfl)
    let first := RomanNumerals.value (String.singleton text.toList[index])
    let e1 := setLocal env "v" (number first)
    have ready1 : Locals e1 globals text index total := ready.value first
    have getFirst : e1 "v" = some (number first) := by simp [e1, setLocal]
    apply BlockReturns.assign (read text index inside valid heap globals table env ready.input ready.table
      (.name "i") (by intros; simp) (Evaluates.lookup ready.position))
    by_cases secondInside : index + 1 < text.toList.length
    · apply BlockReturns.branch (flag := true) (has_second text index total heap globals e1 ready1)
        (by simp only [secondInside]; rfl)
      let second := RomanNumerals.value (String.singleton text.toList[index + 1])
      let e2 := setLocal e1 "v2" (number second)
      have ready2 : Locals e2 globals text index total := ready1.second second
      have getFirst2 : e2 "v" = some (number first) := by simp [e2, e1, setLocal]
      have getSecond : e2 "v2" = some (number second) := by simp [e2, setLocal]
      have equation : suffix text index =
          if first < second then second - first + suffix text (index + 2)
          else first + suffix text (index + 1) := by
        have step := RomanNumerals.parse_drop_pair (text.toList.map String.singleton) index
          (by simpa using secondInside)
        simp only [List.getElem_map] at step
        exact step
      apply BlockReturns.assign (read text (index + 1) secondInside valid heap globals table e1
        ready1.input ready1.table _ (by intros; simp) (next_position e1 heap globals index ready1.position))
      by_cases subtract : first < second
      · apply BlockReturns.branch (flag := true) (increasing e2 heap globals first second getFirst2 getSecond)
          (by simp only [subtract]; rfl)
        apply BlockReturns.augAssign (middle := heap) (afterOperation := heap) ready2.total (value := number (total + (second - first)))
          (right := number (second - first))
        · exact Evaluates.binary (Evaluates.lookup getSecond) (Evaluates.lookup getFirst2)
            (by simp [Mutable.binary, scalarBinary, number, Nat.cast_sub (Nat.le_of_lt subtract)])
        · simp [inplace, Mutable.binary, scalarBinary, number]
        apply BlockReturns.augAssign (middle := heap) (afterOperation := heap) (left := number index) (value := number (index + 2))
        · simpa [setLocal] using ready2.position
        · exact Evaluates.constant
        · simp [inplace, Mutable.binary, scalarBinary, number]
        apply BlockReturns.continue_loop (by rfl)
        have recursive := loop text valid (index + 2) (total + (second - first)) heap globals table _
          (ready2.advance (index + 2) (total + (second - first)))
        simpa only [equation, if_pos subtract, Nat.add_assoc] using recursive
      · apply BlockReturns.branch (flag := false) (increasing e2 heap globals first second getFirst2 getSecond)
          (by simp only [subtract]; rfl)
        apply add_one text index total first heap globals e2 ready2 getFirst2
        have recursive := loop text valid (index + 1) (total + first) heap globals table _
          (ready2.advance (index + 1) (total + first))
        simpa only [equation, if_neg subtract, Nat.add_assoc] using recursive
    · apply BlockReturns.branch (flag := false) (has_second text index total heap globals e1 ready1)
        (by simp only [secondInside]; rfl)
      apply add_one text index total first heap globals e1 ready1 getFirst
      have equation : suffix text index = first := by
        have step := RomanNumerals.parse_drop_single (text.toList.map String.singleton) index
          (by simpa using inside) (by simpa using Nat.le_of_not_gt secondInside)
        simp only [List.getElem_map] at step
        exact step
      have finished : suffix text (index + 1) = 0 := by
        simp only [suffix, List.drop_eq_nil_of_le (show (text.toList.map String.singleton).length ≤ index + 1 by simpa using Nat.le_of_not_gt secondInside), RomanNumerals.parse]
      have recursive := loop text valid (index + 1) (total + first) heap globals table _
        (ready1.advance (index + 1) (total + first))
      simpa only [equation, finished, Nat.add_zero] using recursive
  · have finished : suffix text index = 0 := by
      simp only [suffix, List.drop_eq_nil_of_le (show (text.toList.map String.singleton).length ≤ index by simpa using Nat.le_of_not_gt inside), RomanNumerals.parse]
    rw [finished, Nat.add_zero]
    apply BlockReturns.while_loop (flag := false) (active text index total heap globals env ready)
      (by simp only [inside]; rfl)
    exact BlockReturns.return_value (Evaluates.lookup ready.total)
termination_by text.toList.length - index

decreasing_by all_goals omega

end solvers.«89».roman_to_int
