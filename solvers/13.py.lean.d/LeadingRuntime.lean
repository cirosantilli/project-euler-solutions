import solvers.«13.py.lean.d».FirstDigitsRuntime

namespace solvers.«13».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open HeapCalls.Runtime.NestedIntList (encode)
private abbrev program := Solvers.P13Ast.module

private theorem parsed_prefix (n : Int) (k : Nat) (nonnegative : 0 ≤ n)
    (nonempty : (String.ofList ((toString n).toList.take k)).isEmpty = false) :
    ∃ answer : Int, Calls.integerConversion [.string (String.ofList ((toString n).toList.take k))] =
      some (.ok (.int answer)) := by
  have printed : toString n = toString n.toNat := by
    calc
      toString n = toString (n.toNat : Int) := congrArg (fun value : Int => toString value) (Int.toNat_of_nonneg nonnegative).symm
      _ = toString n.toNat := rfl
  have present : (toString n).toList.take k ≠ [] := by
    intro absent
    rw [absent] at nonempty
    cases nonempty
  have digits : ∀ c ∈ (toString n).toList.take k, c.isDigit := by
    intro c member
    have original : c ∈ Nat.toDigits 10 n.toNat := by
      simpa [printed] using List.mem_of_mem_take member
    exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) original
  rw [Calls.integerConversion, FormalSW.Py.Builtin.Int.parseText_digits _ present digits]
  exact ⟨_, rfl⟩

/-- The final conditional parses only a nonempty prefix, preserving the zero-length case. -/
theorem leading_cost (n : Int) (k size : Nat) (heap : Heap) (env : HeapCalls.Env)
    (nonnegative : 0 ≤ n)
    (lookup : env "leading" = some (.scalar (.string (String.ofList ((toString n).toList.take k)))))
    (length : (String.ofList ((toString n).toList.take k)).length ≤ size+1) :
    ∃ answer : Int, EvaluatesIn .word program env
      (.ifExp (.name "leading") (.builtinCall .int [.name "leading"]) (.constant (.int 0))) heap
      (size+10) (.scalar (.int answer)) heap := by
  let leading := String.ofList ((toString n).toList.take k)
  by_cases empty : leading.isEmpty = true
  · refine ⟨0, ?_⟩
    apply EvaluatesIn.mono (small := 4) (larger := by omega)
    apply EvaluatesIn.conditional (ct := 1) (cb := 1) (flag := false) (value := .string leading)
      (EvaluatesIn.lookup lookup)
    · change some (Except.ok (!leading.isEmpty), heap) = _
      rw [empty]
      rfl
    · exact EvaluatesIn.constant
  · have nonempty : leading.isEmpty = false := by
      cases h : leading.isEmpty with
      | false => rfl
      | true => exact False.elim (empty h)
    obtain ⟨answer, parsed⟩ := parsed_prefix n k nonnegative nonempty
    have parsedRun := EvaluatesIn.parse_string_word
      (module := program) (heap := heap) (EvaluatesIn.lookup lookup) parsed
    refine ⟨answer, ?_⟩
    have conditional := EvaluatesIn.conditional (ct := 1) (flag := true) (no := .constant (.int 0))
      (EvaluatesIn.lookup lookup)
      (show Mutable.truth (.scalar (.string leading)) heap = some (.ok true, heap) from by
        change some (Except.ok (!leading.isEmpty), heap) = _
        rw [nonempty]
        rfl) parsedRun
    exact conditional.mono (by omega)

end solvers.«13».runtimeCost
