import ProjectEulerSolutions.P59

open ProjectEulerSolutions.P59

def main : IO Unit := do
  let text ← IO.FS.readFile "0059_cipher.txt"
  IO.println (solve (parseCipher text))
