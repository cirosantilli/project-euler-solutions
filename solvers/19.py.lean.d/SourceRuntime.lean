import solvers.«19.py.lean.d».MonthRuntime
import lang.py.FormalSW.HeapRuntimeCostSetup

namespace solvers.«19».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
private abbrev program := Solvers.P19Ast.module

private theorem month_loop_cost (remaining : Nat) (month first last year weekday count : Int)
    (endpoint : month + remaining = 13) (heap : Heap) (env : HeapCalls.Env)
    (saved : CountEnv env first last weekday count) (getYear : env "year" = some (.scalar (.int year))) :
    ∃ extra finalEnv, ∃ nextWeekday nextCount : Int,
      BlockContinuesIn .word program env [.rangeStepLoop "month" month 13 1 monthBody []]
        heap (135*remaining+4) finalEnv (heap ++ extra) ∧
      CountEnv finalEnv first last nextWeekday nextCount ∧
      finalEnv "year" = some (.scalar (.int year)) := by
  induction remaining generalizing month weekday count heap env with
  | zero =>
      refine ⟨[], env, weekday, count, ?_, saved, getYear⟩
      simpa using (BlockContinuesIn.range_done (body := monthBody)
        (by simp [Calls.rangeContinues]; omega) (BlockContinuesIn.nil (env := env) (heap := heap)))
  | succ remaining ih =>
      let withMonth := setLocal env "month" (.scalar (.int month))
      have monthSaved : CountEnv withMonth first last weekday count := saved.set_other "month" _ (by decide)
      have monthYear : withMonth "year" = some (.scalar (.int year)) := by simpa [withMonth, setLocal] using getYear
      obtain ⟨extra, middleEnv, middleDay, middleCount, bodyRun, middleSaved, middleYear⟩ :=
        month_body_cost first last year month weekday count heap withMonth monthSaved monthYear rfl
      obtain ⟨later, finalEnv, finalDay, finalCount, tailRun, finalSaved, finalYear⟩ :=
        ih (month+1) middleDay middleCount (by omega) (heap ++ extra) middleEnv middleSaved middleYear
      refine ⟨extra ++ later, finalEnv, finalDay, finalCount, ?_, finalSaved, finalYear⟩
      have active : Calls.rangeContinues month 13 1 = true := by simp [Calls.rangeContinues]; omega
      have run := BlockContinuesIn.range_step (env := env) (name := "month") active
        (by simpa only [monthBody, yearBody, Calls.loopContinuation, List.append_assoc] using bodyRun.append tailRun)
      exact run.mono (by omega)

private theorem year_body_cost (first last year weekday count : Int) (heap : Heap) (env : HeapCalls.Env)
    (saved : CountEnv env first last weekday count) (getYear : env "year" = some (.scalar (.int year))) :
    ∃ extra finalEnv, ∃ nextWeekday nextCount : Int,
      BlockContinuesIn .word program env yearBody heap 1633 finalEnv (heap ++ extra) ∧
      CountEnv finalEnv first last nextWeekday nextCount := by
  obtain ⟨extra, finalEnv, finalDay, finalCount, run, finalSaved, _⟩ :=
    month_loop_cost 12 1 first last year weekday count (by decide) heap env saved getYear
  refine ⟨extra, finalEnv, finalDay, finalCount, ?_, finalSaved⟩
  exact BlockContinuesIn.for_range_pair EvaluatesIn.constant EvaluatesIn.constant run

private theorem year_loop_cost (remaining : Nat) (year first last weekday count : Int)
    (endpoint : remaining = (last+1-year).toNat) (heap : Heap) (env : HeapCalls.Env)
    (saved : CountEnv env first last weekday count) :
    ∃ extra finalEnv, ∃ nextWeekday nextCount : Int,
      BlockContinuesIn .word program env [.rangeStepLoop "year" year (last+1) 1 yearBody []]
        heap (1637*remaining+4) finalEnv (heap ++ extra) ∧
      CountEnv finalEnv first last nextWeekday nextCount := by
  induction remaining generalizing year weekday count heap env with
  | zero =>
      refine ⟨[], env, weekday, count, ?_, saved⟩
      simpa using (BlockContinuesIn.range_done (body := yearBody)
        (by simp [Calls.rangeContinues]; omega) (BlockContinuesIn.nil (env := env) (heap := heap)))
  | succ remaining ih =>
      let withYear := setLocal env "year" (.scalar (.int year))
      have yearSaved : CountEnv withYear first last weekday count := saved.set_other "year" _ (by decide)
      obtain ⟨extra, middleEnv, middleDay, middleCount, bodyRun, middleSaved⟩ :=
        year_body_cost first last year weekday count heap withYear yearSaved rfl
      obtain ⟨later, finalEnv, finalDay, finalCount, tailRun, finalSaved⟩ :=
        ih (year+1) middleDay middleCount (by omega) (heap ++ extra) middleEnv middleSaved
      refine ⟨extra ++ later, finalEnv, finalDay, finalCount, ?_, finalSaved⟩
      have active : Calls.rangeContinues year (last+1) 1 = true := by simp [Calls.rangeContinues]; omega
      have run := BlockContinuesIn.range_step (env := env) (name := "year") active
        (by simpa only [yearBody, Calls.loopContinuation, List.append_assoc] using bodyRun.append tailRun)
      exact run.mono (by omega)

/-- The runtime bound follows the actual source loops, including all twelve months per year. -/
theorem source_bound (first last : Int) :
    TimeBound .word program "solve" [.scalar (.int first), .scalar (.int last)] []
      (1637*(last+1-1900).toNat+32) := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := functionEnv args body [.scalar (.int first), .scalar (.int last)] []
  let withDay := setLocal env "dow_first_of_month" (.scalar (.int 0))
  let initial := setLocal withDay "count" (.scalar (.int 0))
  obtain ⟨extra, finalEnv, finalDay, finalCount, loop, finalSaved⟩ :=
    year_loop_cost (last+1-1900).toNat 1900 first last 0 0 rfl [] initial (by exact ⟨rfl, rfl, rfl, rfl⟩)
  have forLoop := BlockContinuesIn.for_range_pair
    (EvaluatesIn.constant (module := program) (env := initial) (heap := []) (value := .int 1900))
    (EvaluatesIn.binary (op := .add) (work := 1) (av := .int last) (bv := .int 1)
      (EvaluatesIn.lookup (name := "end_year") rfl) EvaluatesIn.constant rfl rfl) loop
  have dayInit := BlockContinuesIn.assign (arithmetic := .word) (module := program) (env := env) (heap := [])
    (name := "dow_first_of_month") (EvaluatesIn.constant (value := .int 0))
  have countInit := BlockContinuesIn.assign (arithmetic := .word) (module := program) (env := withDay) (heap := [])
    (name := "count") (EvaluatesIn.constant (value := .int 0))
  have returned := ((dayInit.append countInit).append forLoop).returns
    (BlockReturnsIn.return_value (following := []) (EvaluatesIn.lookup finalSaved.2.2.2))
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := args) (body := body) (values := [.scalar (.int first), .scalar (.int last)]) rfl rfl returned
  exact called.timeBound_append.mono (by change 2+2+1+(_+_+_+_)+1 ≤ _; omega)

end solvers.«19».runtimeCost
