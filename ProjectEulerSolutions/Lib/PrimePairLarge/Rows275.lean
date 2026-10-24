import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8849_small : List Nat := []
theorem row8849_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8849 row8849_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8849_block_005 : List Nat := [9041, 9419, 10079, 10193]
def plan8849_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 37), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .possible, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .second, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 11), .possible, (.firstFactor 23), .first, .first, (.firstFactor 13), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8849_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8849 row8849_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8849_block_005
  · decide
  · rfl

def row8849_block_006 : List Nat := [10781, 11789, 11939]
def plan8849_block_006 : List Budget.Choice := [.first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 11), .first, (.secondFactor 23), (.firstFactor 31), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8849_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8849 row8849_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8849_block_006
  · decide
  · rfl

def row8849_block_007 : List Nat := [12653, 13331, 14423]
def plan8849_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .possible, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .second, (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 29), (.firstFactor 23), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 23), (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 17), .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 7), .second, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8849_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8849 row8849_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8849_block_007
  · decide
  · rfl

def row8849_block_008 : List Nat := [14633, 15497, 16073]
def plan8849_block_008 : List Budget.Choice := [.first, (.firstFactor 23), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 7), .second, .first, .first, .second, .first, .second, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .possible, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 7), .second, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8849_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8849 row8849_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8849_block_008
  · decide
  · rfl

def row8849_block_009 : List Nat := []
def plan8849_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8849_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8849 row8849_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8849_block_009
  · decide
  · rfl

def row8849_block_010 : List Nat := []
theorem row8849_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8849 row8849_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8849_block_011 : List Nat := []
theorem row8849_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8849 row8849_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8849_block_012 : List Nat := []
theorem row8849_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8849 row8849_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8849_block_013 : List Nat := []
theorem row8849_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8849 row8849_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8849_large_left_left : List Nat := row8849_small ++ row8849_block_005
theorem row8849_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8849 row8849_large_left_left :=
  Budget.RowUpperBound.branch row8849_small_covers row8849_block_005_covers

def row8849_large_left_right_right : List Nat := row8849_block_007 ++ row8849_block_008
theorem row8849_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8849 row8849_large_left_right_right :=
  Budget.RowUpperBound.branch row8849_block_007_covers row8849_block_008_covers

def row8849_large_left_right : List Nat := row8849_block_006 ++ row8849_large_left_right_right
theorem row8849_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8849 row8849_large_left_right :=
  Budget.RowUpperBound.branch row8849_block_006_covers row8849_large_left_right_right_covers

def row8849_large_left : List Nat := row8849_large_left_left ++ row8849_large_left_right
theorem row8849_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8849 row8849_large_left :=
  Budget.RowUpperBound.branch row8849_large_left_left_covers row8849_large_left_right_covers

def row8849_large_right_left : List Nat := row8849_block_009 ++ row8849_block_010
theorem row8849_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8849 row8849_large_right_left :=
  Budget.RowUpperBound.branch row8849_block_009_covers row8849_block_010_covers

def row8849_large_right_right_right : List Nat := row8849_block_012 ++ row8849_block_013
theorem row8849_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8849 row8849_large_right_right_right :=
  Budget.RowUpperBound.branch row8849_block_012_covers row8849_block_013_covers

def row8849_large_right_right : List Nat := row8849_block_011 ++ row8849_large_right_right_right
theorem row8849_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8849 row8849_large_right_right :=
  Budget.RowUpperBound.branch row8849_block_011_covers row8849_large_right_right_right_covers

def row8849_large_right : List Nat := row8849_large_right_left ++ row8849_large_right_right
theorem row8849_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8849 row8849_large_right :=
  Budget.RowUpperBound.branch row8849_large_right_left_covers row8849_large_right_right_covers

def row8849_large : List Nat := row8849_large_left ++ row8849_large_right
theorem row8849_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8849 row8849_large :=
  Budget.RowUpperBound.branch row8849_large_left_covers row8849_large_right_covers

def row8849 : List Nat := row8849_large
theorem row8849_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8849 row8849 := row8849_large_covers

def row8861_small : List Nat := []
theorem row8861_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8861 row8861_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8861_block_005 : List Nat := [9029, 9887, 9941, 10343]
def plan8861_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.firstFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 31), .first, (.firstFactor 37), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row8861_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8861 row8861_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8861_block_005
  · decide
  · rfl

def row8861_block_006 : List Nat := [10781, 11549, 11939, 11969, 12011]
def plan8861_block_006 : List Budget.Choice := [(.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.secondFactor 29), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .possible, .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8861_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8861 row8861_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8861_block_006
  · decide
  · rfl

def row8861_block_007 : List Nat := [13619, 13649, 14087]
def plan8861_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 29), .second, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .second, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, (.secondFactor 19), .first, .possible, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row8861_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8861 row8861_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8861_block_007
  · decide
  · rfl

def row8861_block_008 : List Nat := [14753, 15767, 16229]
def plan8861_block_008 : List Budget.Choice := [.first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 7), .possible, .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .second, .first, (.secondFactor 23), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .possible, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row8861_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8861 row8861_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8861_block_008
  · decide
  · rfl

def row8861_block_009 : List Nat := []
def plan8861_block_009 : List Budget.Choice := [.first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 13), .second, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 17), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8861_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8861 row8861_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8861_block_009
  · decide
  · rfl

def row8861_block_010 : List Nat := []
theorem row8861_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8861 row8861_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8861_block_011 : List Nat := []
theorem row8861_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8861 row8861_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8861_block_012 : List Nat := []
theorem row8861_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8861 row8861_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8861_block_013 : List Nat := []
theorem row8861_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8861 row8861_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8861_large_left_left : List Nat := row8861_small ++ row8861_block_005
theorem row8861_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8861 row8861_large_left_left :=
  Budget.RowUpperBound.branch row8861_small_covers row8861_block_005_covers

def row8861_large_left_right_right : List Nat := row8861_block_007 ++ row8861_block_008
theorem row8861_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8861 row8861_large_left_right_right :=
  Budget.RowUpperBound.branch row8861_block_007_covers row8861_block_008_covers

def row8861_large_left_right : List Nat := row8861_block_006 ++ row8861_large_left_right_right
theorem row8861_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8861 row8861_large_left_right :=
  Budget.RowUpperBound.branch row8861_block_006_covers row8861_large_left_right_right_covers

def row8861_large_left : List Nat := row8861_large_left_left ++ row8861_large_left_right
theorem row8861_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8861 row8861_large_left :=
  Budget.RowUpperBound.branch row8861_large_left_left_covers row8861_large_left_right_covers

def row8861_large_right_left : List Nat := row8861_block_009 ++ row8861_block_010
theorem row8861_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8861 row8861_large_right_left :=
  Budget.RowUpperBound.branch row8861_block_009_covers row8861_block_010_covers

def row8861_large_right_right_right : List Nat := row8861_block_012 ++ row8861_block_013
theorem row8861_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8861 row8861_large_right_right_right :=
  Budget.RowUpperBound.branch row8861_block_012_covers row8861_block_013_covers

def row8861_large_right_right : List Nat := row8861_block_011 ++ row8861_large_right_right_right
theorem row8861_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8861 row8861_large_right_right :=
  Budget.RowUpperBound.branch row8861_block_011_covers row8861_large_right_right_right_covers

def row8861_large_right : List Nat := row8861_large_right_left ++ row8861_large_right_right
theorem row8861_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8861 row8861_large_right :=
  Budget.RowUpperBound.branch row8861_large_right_left_covers row8861_large_right_right_covers

def row8861_large : List Nat := row8861_large_left ++ row8861_large_right
theorem row8861_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8861 row8861_large :=
  Budget.RowUpperBound.branch row8861_large_left_covers row8861_large_right_covers

def row8861 : List Nat := row8861_large
theorem row8861_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8861 row8861 := row8861_large_covers

def row8863_small : List Nat := []
theorem row8863_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8863 row8863_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8863_block_005 : List Nat := [8887, 9277, 9349, 9871]
def plan8863_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 29), .first, .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .second, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8863_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8863 row8863_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8863_block_005
  · decide
  · rfl

def row8863_block_006 : List Nat := [10753, 11131, 11299, 11827]
def plan8863_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .possible, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .second, .possible, (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .second, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 17), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8863_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8863 row8863_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8863_block_006
  · decide
  · rfl

def row8863_block_007 : List Nat := [12919, 13177, 14437]
def plan8863_block_007 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), (.secondFactor 23), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .possible, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row8863_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8863 row8863_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8863_block_007
  · decide
  · rfl

def row8863_block_008 : List Nat := [14629, 14851, 15277]
def plan8863_block_008 : List Budget.Choice := [(.secondFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .second, .first, .first, .second, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .possible, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 13), .second, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 29), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 31), (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8863_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8863 row8863_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8863_block_008
  · decide
  · rfl

def row8863_block_009 : List Nat := []
def plan8863_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 31), (.firstFactor 17), (.secondFactor 23), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .second, (.secondFactor 23), (.secondFactor 19), (.firstFactor 37), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8863_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8863 row8863_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8863_block_009
  · decide
  · rfl

def row8863_block_010 : List Nat := []
theorem row8863_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8863 row8863_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8863_block_011 : List Nat := []
theorem row8863_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8863 row8863_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8863_block_012 : List Nat := []
theorem row8863_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8863 row8863_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8863_block_013 : List Nat := []
theorem row8863_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8863 row8863_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8863_large_left_left : List Nat := row8863_small ++ row8863_block_005
theorem row8863_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8863 row8863_large_left_left :=
  Budget.RowUpperBound.branch row8863_small_covers row8863_block_005_covers

def row8863_large_left_right_right : List Nat := row8863_block_007 ++ row8863_block_008
theorem row8863_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8863 row8863_large_left_right_right :=
  Budget.RowUpperBound.branch row8863_block_007_covers row8863_block_008_covers

def row8863_large_left_right : List Nat := row8863_block_006 ++ row8863_large_left_right_right
theorem row8863_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8863 row8863_large_left_right :=
  Budget.RowUpperBound.branch row8863_block_006_covers row8863_large_left_right_right_covers

def row8863_large_left : List Nat := row8863_large_left_left ++ row8863_large_left_right
theorem row8863_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8863 row8863_large_left :=
  Budget.RowUpperBound.branch row8863_large_left_left_covers row8863_large_left_right_covers

def row8863_large_right_left : List Nat := row8863_block_009 ++ row8863_block_010
theorem row8863_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8863 row8863_large_right_left :=
  Budget.RowUpperBound.branch row8863_block_009_covers row8863_block_010_covers

def row8863_large_right_right_right : List Nat := row8863_block_012 ++ row8863_block_013
theorem row8863_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8863 row8863_large_right_right_right :=
  Budget.RowUpperBound.branch row8863_block_012_covers row8863_block_013_covers

def row8863_large_right_right : List Nat := row8863_block_011 ++ row8863_large_right_right_right
theorem row8863_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8863 row8863_large_right_right :=
  Budget.RowUpperBound.branch row8863_block_011_covers row8863_large_right_right_right_covers

def row8863_large_right : List Nat := row8863_large_right_left ++ row8863_large_right_right
theorem row8863_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8863 row8863_large_right :=
  Budget.RowUpperBound.branch row8863_large_right_left_covers row8863_large_right_right_covers

def row8863_large : List Nat := row8863_large_left ++ row8863_large_right
theorem row8863_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8863 row8863_large :=
  Budget.RowUpperBound.branch row8863_large_left_covers row8863_large_right_covers

def row8863 : List Nat := row8863_large
theorem row8863_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8863 row8863 := row8863_large_covers

def row8867_small : List Nat := []
theorem row8867_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8867 row8867_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8867_block_005 : List Nat := [9323, 10247]
def plan8867_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 17), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 7), .first, .possible, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8867_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8867 row8867_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8867_block_005
  · decide
  · rfl

def row8867_block_006 : List Nat := [11909, 11927, 12269]
def plan8867_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, .possible, (.secondFactor 19), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 13), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8867_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8867 row8867_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8867_block_006
  · decide
  · rfl

def row8867_block_007 : List Nat := [12923, 13007, 13649]
def plan8867_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 23), .second, .first, .first, .first, (.firstFactor 37), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .second, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), .first, (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8867_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8867 row8867_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8867_block_007
  · decide
  · rfl

def row8867_block_008 : List Nat := [15737, 16427]
def plan8867_block_008 : List Budget.Choice := [.first, (.secondFactor 31), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.firstFactor 37), (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8867_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8867 row8867_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8867_block_008
  · decide
  · rfl

def row8867_block_009 : List Nat := [16661, 16703]
def plan8867_block_009 : List Budget.Choice := [.first, (.firstFactor 19), (.secondFactor 17), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 29), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8867_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8867 row8867_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8867_block_009
  · decide
  · rfl

def row8867_block_010 : List Nat := []
theorem row8867_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8867 row8867_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8867_block_011 : List Nat := []
theorem row8867_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8867 row8867_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8867_block_012 : List Nat := []
theorem row8867_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8867 row8867_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8867_block_013 : List Nat := []
theorem row8867_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8867 row8867_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8867_large_left_left : List Nat := row8867_small ++ row8867_block_005
theorem row8867_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8867 row8867_large_left_left :=
  Budget.RowUpperBound.branch row8867_small_covers row8867_block_005_covers

def row8867_large_left_right_right : List Nat := row8867_block_007 ++ row8867_block_008
theorem row8867_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8867 row8867_large_left_right_right :=
  Budget.RowUpperBound.branch row8867_block_007_covers row8867_block_008_covers

def row8867_large_left_right : List Nat := row8867_block_006 ++ row8867_large_left_right_right
theorem row8867_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8867 row8867_large_left_right :=
  Budget.RowUpperBound.branch row8867_block_006_covers row8867_large_left_right_right_covers

def row8867_large_left : List Nat := row8867_large_left_left ++ row8867_large_left_right
theorem row8867_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8867 row8867_large_left :=
  Budget.RowUpperBound.branch row8867_large_left_left_covers row8867_large_left_right_covers

def row8867_large_right_left : List Nat := row8867_block_009 ++ row8867_block_010
theorem row8867_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8867 row8867_large_right_left :=
  Budget.RowUpperBound.branch row8867_block_009_covers row8867_block_010_covers

def row8867_large_right_right_right : List Nat := row8867_block_012 ++ row8867_block_013
theorem row8867_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8867 row8867_large_right_right_right :=
  Budget.RowUpperBound.branch row8867_block_012_covers row8867_block_013_covers

def row8867_large_right_right : List Nat := row8867_block_011 ++ row8867_large_right_right_right
theorem row8867_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8867 row8867_large_right_right :=
  Budget.RowUpperBound.branch row8867_block_011_covers row8867_large_right_right_right_covers

def row8867_large_right : List Nat := row8867_large_right_left ++ row8867_large_right_right
theorem row8867_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8867 row8867_large_right :=
  Budget.RowUpperBound.branch row8867_large_right_left_covers row8867_large_right_right_covers

def row8867_large : List Nat := row8867_large_left ++ row8867_large_right
theorem row8867_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8867 row8867_large :=
  Budget.RowUpperBound.branch row8867_large_left_covers row8867_large_right_covers

def row8867 : List Nat := row8867_large
theorem row8867_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8867 row8867 := row8867_large_covers

end ProjectEulerSolutions.PrimePairLarge
