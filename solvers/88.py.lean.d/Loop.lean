import solvers.«88.py.lean.d».Iteration

namespace solvers.«88»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

private theorem range_more (next last : Nat) (active : next ≤ last) :
    Calls.rangeContinues next ((last + 1 : Nat) : Int) 1 = true := by
  simp [Calls.rangeContinues]
  omega

private theorem range_finished (next last : Nat) (finished : last < next) :
    Calls.rangeContinues next ((last + 1 : Nat) : Int) 1 = false := by
  simp [Calls.rangeContinues]
  omega

/-- The actual range loop visits each remaining factor without changing the captured cells. -/
theorem loop (maxSize limit : Nat) (before : List Nat) (valid : Admissible before)
    (factor : Nat) (lower : 2 ≤ factor) (context : Context) (cells : Cells context maxSize limit)
    (descendants : Descendants maxSize limit before context)
    (best : List Nat) (length : best.length = maxSize + 1)
    (env : HeapCalls.Env) (ready : Locals env context before) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "f" factor ((limit / before.prod + 1 : Nat) : Int) 1 dfsBody []]
      (working context best) finalEnv
      (working context (accumulate (search maxSize limit factor before.prod before.sum before.length) best)) ∧
      Locals finalEnv context before := by
  have positive := product_positive before valid
  by_cases active : factor ≤ limit / before.prod
  · have inside : before.prod * factor ≤ limit := by
      simpa only [Nat.mul_comm] using (Nat.le_div_iff_mul_le (by omega : 0 < before.prod)).mp active
    let entered := setLocal env "f" (number factor)
    let measured := measuredEnv entered before factor
    let updated := accumulate (branch maxSize limit before.prod before.sum before.length factor) best
    have enteredReady := ready.factor factor
    have measuredReady := enteredReady.measured factor
    obtain ⟨finalEnv, continued, finalReady⟩ := loop maxSize limit before valid (factor + 1) (by omega)
      context cells descendants updated (by simpa [updated, accumulate_length] using length) measured measuredReady
    refine ⟨finalEnv, ?_, finalReady⟩
    rw [accumulate_step maxSize limit factor before.prod before.sum before.length best (by omega) lower active]
    apply BlockContinues.range_step (range_more factor (limit / before.prod) active)
    have iterated := iteration maxSize limit before valid factor lower inside context cells descendants
      best length entered enteredReady (by simp [entered, setLocal])
    simpa only [Calls.loopContinuation, show Calls.loopControl dfsBody = false from rfl, if_false,
      Nat.cast_add, Nat.cast_one] using iterated.append continued
  · refine ⟨env, ?_, ready⟩
    rw [search_done maxSize limit factor before.prod before.sum before.length (by omega) lower (by omega)]
    apply BlockContinues.range_done (range_finished factor (limit / before.prod) (by omega))
    exact .nil
termination_by limit / before.prod + 1 - factor
decreasing_by omega

end solvers.«88»
