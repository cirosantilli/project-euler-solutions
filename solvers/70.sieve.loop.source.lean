import solvers.«70.sieve.pass.source»

/-! Both early breaks and every live-list iteration of the inner linear sieve. -/
namespace solvers.«70».sieve
open FormalSW.PyAst HeapCalls ProjectEulerStatements.TotientLinearSieve
set_option Elab.async false

/-- Every remaining prime follows the exact source trace of the mathematical pass. -/
theorem pass_loop_result (limit i : Nat) (cache remaining : List Nat) (table : Table)
    (position : Nat) (suffix : cache.drop position = remaining)
    (primes : ∀ p ∈ cache, Nat.Prime p) (within : i ≤ limit)
    (base : Mutable.Heap) (env : HeapCalls.Env) (ready : Ready limit i base.length env) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "p" (sequenceSource (.ref (base.length + 4))) position innerBody []]
      (realize base (limit + 1) table cache) finalEnv
      (realize base (limit + 1) (pass limit i remaining table) cache) ∧
      Ready limit i base.length finalEnv := by
  have sequence : Mutable.sequence (.ref (base.length + 4))
      (realize base (limit + 1) table cache) =
        some (.ok (cache.map number), realize base (limit + 1) table cache) := by
    simp [Mutable.sequence, cache_found]
  induction remaining generalizing position table env with
  | nil =>
      have missing : cache[position]? = none := by
        simpa only [suffix, List.head?_nil] using
          (List.head?_drop (l := cache) (i := position)).symm
      refine ⟨env, ?_, ready⟩
      apply BlockContinues.sequence_done
        (source := sequenceSource (.ref (base.length + 4))) (items := cache.map number) sequence
        (by simp [List.getElem?_map, missing])
      exact .nil
  | cons p rest ih =>
      have item : cache[position]? = some p := by
        simpa only [suffix, List.head?_cons] using
          (List.head?_drop (l := cache) (i := position)).symm
      obtain ⟨bounded, equal⟩ := List.getElem?_eq_some_iff.mp item
      have prime := primes p (equal ▸ List.getElem_mem bounded)
      let indexed := setLocal env "p" (number p)
      let multiplied := setLocal indexed "ip" (number (i * p))
      have indexedReady : Ready limit i base.length indexed := ready.set_p p
      have multipliedReady : Ready limit i base.length multiplied := indexedReady.set_ip (i * p)
      have getP : multiplied "p" = some (number p) := by simp [multiplied, indexed, setLocal]
      have getIP : multiplied "ip" = some (number (i * p)) := rfl
      have assigned : BlockContinues program indexed [.assign [.name "ip"] productExpression]
          (realize base (limit + 1) table cache) multiplied
          (realize base (limit + 1) table cache) :=
        BlockContinues.assign (product_result i p _ indexed indexedReady.getI rfl)
      have compared := bound_result limit (i * p) (realize base (limit + 1) table cache)
        multiplied multipliedReady.getLimit getIP
      by_cases outside : limit < i * p
      · refine ⟨multiplied, ?_, multipliedReady⟩
        simp only [pass, if_pos outside]
        apply BlockContinues.sequence_step
            (source := sequenceSource (.ref (base.length + 4)))
            (items := cache.map number) (value := number p)
          sequence (by simp [List.getElem?_map, item])
        rw [inner_source]
        apply assigned.append
        apply BlockContinues.branch_next (flag := true)
          (by simpa only [outside, decide_true] using compared) rfl
        exact BlockContinues.break_loop rfl .nil
      · have productBound : i * p < limit + 1 := by omega
        let marked := Function.update (fun n => (table n).2) (i * p) true
        have markedSource := set_mark base (limit + 1) (i * p) (fun n => (table n).1)
          (fun n => (table n).2) cache multiplied productBound multipliedReady.getMarks getIP
        have dividesTest := divides_result i p prime
          (working base (limit + 1) (fun n => (table n).1) marked cache)
          multiplied multipliedReady.getI getP
        by_cases divides : i % p = 0
        · have stored := set_phi base (limit + 1) (i * p) ((table i).1 * p)
            (fun n => (table n).1) marked cache multiplied "ip" repeatedExpression productBound
            multipliedReady.getPhi getIP
            (repeated_result limit i p table base marked cache multiplied within
              multipliedReady getP)
          have written : working base (limit + 1)
              (Function.update (fun n => (table n).1) (i * p) ((table i).1 * p)) marked cache =
                realize base (limit + 1) (write i p table) cache := by
            simpa only [writtenValue, divides, if_true, marked] using
              write_heap base (limit + 1) i p table cache
          rw [written] at stored
          refine ⟨multiplied, ?_, multipliedReady⟩
          simp only [pass, if_neg outside, if_pos divides]
          apply BlockContinues.sequence_step
            (source := sequenceSource (.ref (base.length + 4)))
            (items := cache.map number) (value := number p)
            sequence (by simp [List.getElem?_map, item])
          rw [inner_source]
          apply assigned.append
          apply BlockContinues.branch_next (flag := false)
            (by simpa only [outside, decide_false] using compared) rfl
          apply markedSource.append
          apply BlockContinues.branch_next (flag := true)
            (by simpa only [divides, decide_true] using dividesTest) rfl
          apply stored.append
          exact BlockContinues.break_loop rfl .nil
        · have stored := set_phi base (limit + 1) (i * p) ((table i).1 * (p - 1))
            (fun n => (table n).1) marked cache multiplied "ip" freshExpression productBound
            multipliedReady.getPhi getIP
            (fresh_result limit i p prime table base marked cache multiplied within
              multipliedReady getP)
          have written : working base (limit + 1)
              (Function.update (fun n => (table n).1) (i * p) ((table i).1 * (p - 1)))
              marked cache =
                realize base (limit + 1) (write i p table) cache := by
            simpa only [writtenValue, divides, if_false, marked] using
              write_heap base (limit + 1) i p table cache
          rw [written] at stored
          obtain ⟨finalEnv, continued, finalReady⟩ := ih (write i p table) (position + 1)
            (by simp [List.drop_add_one_eq_tail_drop, suffix]) multiplied multipliedReady
            (by simp [Mutable.sequence, cache_found])
          refine ⟨finalEnv, ?_, finalReady⟩
          simp only [pass, if_neg outside, if_neg divides]
          apply BlockContinues.sequence_step
            (source := sequenceSource (.ref (base.length + 4)))
            (items := cache.map number) (value := number p)
            sequence (by simp [List.getElem?_map, item])
          rw [inner_source]
          apply assigned.append
          apply BlockContinues.branch_next (flag := false)
            (by simpa only [outside, decide_false] using compared) rfl
          apply markedSource.append
          apply BlockContinues.branch_next (flag := false)
            (by simpa only [divides, decide_false] using dividesTest) rfl
          apply stored.append
          exact BlockContinues.loop_resume continued

theorem pass_result (limit i : Nat) (cache : List Nat) (table : Table)
    (primes : ∀ p ∈ cache, Nat.Prime p) (within : i ≤ limit)
    (base : Mutable.Heap) (env : HeapCalls.Env) (ready : Ready limit i base.length env) :
    ∃ finalEnv, BlockContinues program env [innerLoop]
      (realize base (limit + 1) table cache) finalEnv
      (realize base (limit + 1) (pass limit i cache table) cache) ∧
      Ready limit i base.length finalEnv := by
  obtain ⟨finalEnv, continued, finalReady⟩ :=
    pass_loop_result limit i cache cache table 0 rfl primes within base env ready
  refine ⟨finalEnv, ?_, finalReady⟩
  apply BlockContinues.for_sequence (items := cache.map number)
    (by intro expressions impossible; cases impossible) (Evaluates.lookup ready.getCache)
    (by simp [Mutable.sequence, cache_found]) continued

end solvers.«70».sieve
