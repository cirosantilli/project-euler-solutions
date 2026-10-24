import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10909_small : List Nat := []
theorem row10909_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10909 row10909_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10909_block_005 : List Nat := []
theorem row10909_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10909 row10909_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10909_block_006 : List Nat := [11257]
def plan10909_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 31), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 31), .first, .second, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 13), .possible, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 23), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10909_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10909 row10909_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10909_block_006
  · decide
  · rfl

def row10909_block_007 : List Nat := [12841, 13597, 13687, 14437]
def plan10909_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 19), .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 31), .possible, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .possible, .first, (.firstFactor 23), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10909_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10909 row10909_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10909_block_007
  · decide
  · rfl

def row10909_block_008 : List Nat := [14947]
def plan10909_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10909_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10909 row10909_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10909_block_008
  · decide
  · rfl

def row10909_block_009 : List Nat := []
theorem row10909_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10909 row10909_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10909_block_010 : List Nat := []
theorem row10909_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10909 row10909_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10909_block_011 : List Nat := []
theorem row10909_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10909 row10909_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10909_block_012 : List Nat := []
theorem row10909_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10909 row10909_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10909_block_013 : List Nat := []
theorem row10909_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10909 row10909_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10909_large_left_left : List Nat := row10909_small ++ row10909_block_005
theorem row10909_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10909 row10909_large_left_left :=
  Budget.RowUpperBound.branch row10909_small_covers row10909_block_005_covers

def row10909_large_left_right_right : List Nat := row10909_block_007 ++ row10909_block_008
theorem row10909_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10909 row10909_large_left_right_right :=
  Budget.RowUpperBound.branch row10909_block_007_covers row10909_block_008_covers

def row10909_large_left_right : List Nat := row10909_block_006 ++ row10909_large_left_right_right
theorem row10909_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10909 row10909_large_left_right :=
  Budget.RowUpperBound.branch row10909_block_006_covers row10909_large_left_right_right_covers

def row10909_large_left : List Nat := row10909_large_left_left ++ row10909_large_left_right
theorem row10909_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10909 row10909_large_left :=
  Budget.RowUpperBound.branch row10909_large_left_left_covers row10909_large_left_right_covers

def row10909_large_right_left : List Nat := row10909_block_009 ++ row10909_block_010
theorem row10909_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10909 row10909_large_right_left :=
  Budget.RowUpperBound.branch row10909_block_009_covers row10909_block_010_covers

def row10909_large_right_right_right : List Nat := row10909_block_012 ++ row10909_block_013
theorem row10909_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10909 row10909_large_right_right_right :=
  Budget.RowUpperBound.branch row10909_block_012_covers row10909_block_013_covers

def row10909_large_right_right : List Nat := row10909_block_011 ++ row10909_large_right_right_right
theorem row10909_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10909 row10909_large_right_right :=
  Budget.RowUpperBound.branch row10909_block_011_covers row10909_large_right_right_right_covers

def row10909_large_right : List Nat := row10909_large_right_left ++ row10909_large_right_right
theorem row10909_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10909 row10909_large_right :=
  Budget.RowUpperBound.branch row10909_large_right_left_covers row10909_large_right_right_covers

def row10909_large : List Nat := row10909_large_left ++ row10909_large_right
theorem row10909_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10909 row10909_large :=
  Budget.RowUpperBound.branch row10909_large_left_covers row10909_large_right_covers

def row10909 : List Nat := row10909_large
theorem row10909_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10909 row10909 := row10909_large_covers

def row10937_small : List Nat := []
theorem row10937_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10937 row10937_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10937_block_005 : List Nat := []
theorem row10937_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10937 row10937_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10937_block_006 : List Nat := [11057, 11447, 11987]
def plan10937_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .second, .first, .second, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 31), (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .second, .possible, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row10937_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10937 row10937_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10937_block_006
  · decide
  · rfl

def row10937_block_007 : List Nat := [12959, 13679, 13997]
def plan10937_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 31), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 23), (.secondFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.secondFactor 31), (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .possible, .first, (.firstFactor 23), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10937_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10937 row10937_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10937_block_007
  · decide
  · rfl

def row10937_block_008 : List Nat := [14549, 14639, 14969]
def plan10937_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 29), .possible, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10937_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10937 row10937_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10937_block_008
  · decide
  · rfl

def row10937_block_009 : List Nat := []
theorem row10937_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10937 row10937_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10937_block_010 : List Nat := []
theorem row10937_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10937 row10937_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10937_block_011 : List Nat := []
theorem row10937_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10937 row10937_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10937_block_012 : List Nat := []
theorem row10937_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10937 row10937_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10937_block_013 : List Nat := []
theorem row10937_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10937 row10937_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10937_large_left_left : List Nat := row10937_small ++ row10937_block_005
theorem row10937_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10937 row10937_large_left_left :=
  Budget.RowUpperBound.branch row10937_small_covers row10937_block_005_covers

def row10937_large_left_right_right : List Nat := row10937_block_007 ++ row10937_block_008
theorem row10937_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10937 row10937_large_left_right_right :=
  Budget.RowUpperBound.branch row10937_block_007_covers row10937_block_008_covers

def row10937_large_left_right : List Nat := row10937_block_006 ++ row10937_large_left_right_right
theorem row10937_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10937 row10937_large_left_right :=
  Budget.RowUpperBound.branch row10937_block_006_covers row10937_large_left_right_right_covers

def row10937_large_left : List Nat := row10937_large_left_left ++ row10937_large_left_right
theorem row10937_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10937 row10937_large_left :=
  Budget.RowUpperBound.branch row10937_large_left_left_covers row10937_large_left_right_covers

def row10937_large_right_left : List Nat := row10937_block_009 ++ row10937_block_010
theorem row10937_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10937 row10937_large_right_left :=
  Budget.RowUpperBound.branch row10937_block_009_covers row10937_block_010_covers

def row10937_large_right_right_right : List Nat := row10937_block_012 ++ row10937_block_013
theorem row10937_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10937 row10937_large_right_right_right :=
  Budget.RowUpperBound.branch row10937_block_012_covers row10937_block_013_covers

def row10937_large_right_right : List Nat := row10937_block_011 ++ row10937_large_right_right_right
theorem row10937_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10937 row10937_large_right_right :=
  Budget.RowUpperBound.branch row10937_block_011_covers row10937_large_right_right_right_covers

def row10937_large_right : List Nat := row10937_large_right_left ++ row10937_large_right_right
theorem row10937_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10937 row10937_large_right :=
  Budget.RowUpperBound.branch row10937_large_right_left_covers row10937_large_right_right_covers

def row10937_large : List Nat := row10937_large_left ++ row10937_large_right
theorem row10937_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10937 row10937_large :=
  Budget.RowUpperBound.branch row10937_large_left_covers row10937_large_right_covers

def row10937 : List Nat := row10937_large
theorem row10937_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10937 row10937 := row10937_large_covers

def row10939_small : List Nat := []
theorem row10939_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10939 row10939_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10939_block_005 : List Nat := []
theorem row10939_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10939 row10939_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10939_block_006 : List Nat := [11131, 11173, 11827]
def plan10939_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 37), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 11), .second, .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 19), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .second, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10939_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10939 row10939_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10939_block_006
  · decide
  · rfl

def row10939_block_007 : List Nat := [12577, 12601, 13267, 13591, 13903]
def plan10939_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 17), (.firstFactor 7), .possible, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 13), (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), .second, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .possible, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 37), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10939_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10939 row10939_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10939_block_007
  · decide
  · rfl

def row10939_block_008 : List Nat := [14557]
def plan10939_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.secondFactor 23), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 31), .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10939_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10939 row10939_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10939_block_008
  · decide
  · rfl

def row10939_block_009 : List Nat := []
theorem row10939_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10939 row10939_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10939_block_010 : List Nat := []
theorem row10939_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10939 row10939_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10939_block_011 : List Nat := []
theorem row10939_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10939 row10939_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10939_block_012 : List Nat := []
theorem row10939_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10939 row10939_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10939_block_013 : List Nat := []
theorem row10939_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10939 row10939_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10939_large_left_left : List Nat := row10939_small ++ row10939_block_005
theorem row10939_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10939 row10939_large_left_left :=
  Budget.RowUpperBound.branch row10939_small_covers row10939_block_005_covers

def row10939_large_left_right_right : List Nat := row10939_block_007 ++ row10939_block_008
theorem row10939_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10939 row10939_large_left_right_right :=
  Budget.RowUpperBound.branch row10939_block_007_covers row10939_block_008_covers

def row10939_large_left_right : List Nat := row10939_block_006 ++ row10939_large_left_right_right
theorem row10939_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10939 row10939_large_left_right :=
  Budget.RowUpperBound.branch row10939_block_006_covers row10939_large_left_right_right_covers

def row10939_large_left : List Nat := row10939_large_left_left ++ row10939_large_left_right
theorem row10939_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10939 row10939_large_left :=
  Budget.RowUpperBound.branch row10939_large_left_left_covers row10939_large_left_right_covers

def row10939_large_right_left : List Nat := row10939_block_009 ++ row10939_block_010
theorem row10939_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10939 row10939_large_right_left :=
  Budget.RowUpperBound.branch row10939_block_009_covers row10939_block_010_covers

def row10939_large_right_right_right : List Nat := row10939_block_012 ++ row10939_block_013
theorem row10939_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10939 row10939_large_right_right_right :=
  Budget.RowUpperBound.branch row10939_block_012_covers row10939_block_013_covers

def row10939_large_right_right : List Nat := row10939_block_011 ++ row10939_large_right_right_right
theorem row10939_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10939 row10939_large_right_right :=
  Budget.RowUpperBound.branch row10939_block_011_covers row10939_large_right_right_right_covers

def row10939_large_right : List Nat := row10939_large_right_left ++ row10939_large_right_right
theorem row10939_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10939 row10939_large_right :=
  Budget.RowUpperBound.branch row10939_large_right_left_covers row10939_large_right_right_covers

def row10939_large : List Nat := row10939_large_left ++ row10939_large_right
theorem row10939_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10939 row10939_large :=
  Budget.RowUpperBound.branch row10939_large_left_covers row10939_large_right_covers

def row10939 : List Nat := row10939_large
theorem row10939_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10939 row10939 := row10939_large_covers

def row10949_small : List Nat := []
theorem row10949_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10949 row10949_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10949_block_005 : List Nat := []
theorem row10949_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10949 row10949_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10949_block_006 : List Nat := [11369, 11399, 11903]
def plan10949_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 37), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .possible, .first, (.secondFactor 7), .possible, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10949_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10949 row10949_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10949_block_006
  · decide
  · rfl

def row10949_block_007 : List Nat := [12641, 12809, 13109, 13499]
def plan10949_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .possible, (.secondFactor 37), (.firstFactor 31), (.firstFactor 17), (.secondFactor 13), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 17), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 37), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row10949_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10949 row10949_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10949_block_007
  · decide
  · rfl

def row10949_block_008 : List Nat := []
def plan10949_block_008 : List Budget.Choice := [.first, (.secondFactor 19), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10949_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10949 row10949_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10949_block_008
  · decide
  · rfl

def row10949_block_009 : List Nat := []
theorem row10949_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10949 row10949_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10949_block_010 : List Nat := []
theorem row10949_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10949 row10949_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10949_block_011 : List Nat := []
theorem row10949_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10949 row10949_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10949_block_012 : List Nat := []
theorem row10949_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10949 row10949_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10949_block_013 : List Nat := []
theorem row10949_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10949 row10949_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10949_large_left_left : List Nat := row10949_small ++ row10949_block_005
theorem row10949_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10949 row10949_large_left_left :=
  Budget.RowUpperBound.branch row10949_small_covers row10949_block_005_covers

def row10949_large_left_right_right : List Nat := row10949_block_007 ++ row10949_block_008
theorem row10949_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10949 row10949_large_left_right_right :=
  Budget.RowUpperBound.branch row10949_block_007_covers row10949_block_008_covers

def row10949_large_left_right : List Nat := row10949_block_006 ++ row10949_large_left_right_right
theorem row10949_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10949 row10949_large_left_right :=
  Budget.RowUpperBound.branch row10949_block_006_covers row10949_large_left_right_right_covers

def row10949_large_left : List Nat := row10949_large_left_left ++ row10949_large_left_right
theorem row10949_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10949 row10949_large_left :=
  Budget.RowUpperBound.branch row10949_large_left_left_covers row10949_large_left_right_covers

def row10949_large_right_left : List Nat := row10949_block_009 ++ row10949_block_010
theorem row10949_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10949 row10949_large_right_left :=
  Budget.RowUpperBound.branch row10949_block_009_covers row10949_block_010_covers

def row10949_large_right_right_right : List Nat := row10949_block_012 ++ row10949_block_013
theorem row10949_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10949 row10949_large_right_right_right :=
  Budget.RowUpperBound.branch row10949_block_012_covers row10949_block_013_covers

def row10949_large_right_right : List Nat := row10949_block_011 ++ row10949_large_right_right_right
theorem row10949_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10949 row10949_large_right_right :=
  Budget.RowUpperBound.branch row10949_block_011_covers row10949_large_right_right_right_covers

def row10949_large_right : List Nat := row10949_large_right_left ++ row10949_large_right_right
theorem row10949_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10949 row10949_large_right :=
  Budget.RowUpperBound.branch row10949_large_right_left_covers row10949_large_right_right_covers

def row10949_large : List Nat := row10949_large_left ++ row10949_large_right
theorem row10949_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10949 row10949_large :=
  Budget.RowUpperBound.branch row10949_large_left_covers row10949_large_right_covers

def row10949 : List Nat := row10949_large
theorem row10949_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10949 row10949 := row10949_large_covers

end ProjectEulerSolutions.PrimePairLarge
