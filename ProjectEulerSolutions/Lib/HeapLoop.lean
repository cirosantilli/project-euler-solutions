import lang.py.FormalSW.HeapCalls

/-! Composing loop bodies with their actual `continue` boundary. -/
namespace ProjectEulerSolutions.HeapLoop
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
variable {module : Module} {globals : Mutable.Globals}
variable {env middleEnv finalEnv : HeapCalls.Env} {body tail yes no : List Stmt}
variable {heap middleHeap finalHeap middle : Mutable.Heap} {test : Expr} {value : Mutable.Value}

def Continues (module : Module) (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap) (globals : Mutable.Globals := []) : Prop :=
  ∀ next following anchors stack, Reaches module anchors
    ⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩
    ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ globals

theorem nil (module : Module) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    Continues module env [] heap env heap globals := fun _ _ _ _ => .next rfl .refl

theorem prepend (first : BlockContinues module env body heap middleEnv middleHeap globals)
    (rest : Continues module middleEnv tail middleHeap finalEnv finalHeap globals) :
    Continues module env (body ++ tail) heap finalEnv finalHeap globals := by
  intro next following anchors stack
  have joined :=
    (first (tail ++ [.loopResume next] ++ following) anchors stack).trans (rest next following anchors stack)
  simpa only [List.append_assoc] using joined

theorem branch (flag : Bool) (evaluated : Evaluates module env test heap value middle globals)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : Continues module env ((if flag then yes else no) ++ body) middle finalEnv finalHeap globals) :
    Continues module env (.ifThen test yes no :: body) heap finalEnv finalHeap globals := by
  intro next following anchors stack
  have resumed : Reaches module anchors
      ⟨.block ((if flag then yes else no) ++ (body ++ [.loopResume next] ++ following)) env, stack, middle⟩
      ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ globals := by
    have executed := rest next following anchors stack
    simpa only [List.append_assoc] using executed
  exact .next rfl ((evaluated anchors _).trans (.next (by simp [step, run, truth]) resumed))

theorem skip (module : Module) (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (target : ∀ next following, Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following))
    (globals : Mutable.Globals := []) :
    Continues module env (.continueLoop :: body) heap env heap globals := by
  intro next following anchors stack
  have selected : Calls.loopTarget (body ++ .loopResume next :: following) = some (next, following) := by
    simpa using target next following
  exact .next (by simp [step, selected]) .refl

end ProjectEulerSolutions.HeapLoop
