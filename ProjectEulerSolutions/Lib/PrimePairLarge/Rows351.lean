import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11719_small : List Nat := []
theorem row11719_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11719 row11719_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11719_block_005 : List Nat := []
theorem row11719_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11719 row11719_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11719_block_006 : List Nat := [11821, 11971]
def plan11719_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row11719_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11719 row11719_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11719_block_006
  · decide
  · rfl

def row11719_block_007 : List Nat := [12619]
def plan11719_block_007 : List Budget.Choice := [(.secondFactor 19), (.firstFactor 37), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 13), .second, .first, (.secondFactor 17), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.secondFactor 31), (.firstFactor 11), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .second, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11719_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11719 row11719_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11719_block_007
  · decide
  · rfl

def row11719_block_008 : List Nat := []
theorem row11719_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11719 row11719_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11719_block_009 : List Nat := []
theorem row11719_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11719 row11719_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11719_block_010 : List Nat := []
theorem row11719_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11719 row11719_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11719_block_011 : List Nat := []
theorem row11719_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11719 row11719_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11719_block_012 : List Nat := []
theorem row11719_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11719 row11719_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11719_block_013 : List Nat := []
theorem row11719_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11719 row11719_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11719_large_left_left : List Nat := row11719_small ++ row11719_block_005
theorem row11719_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11719 row11719_large_left_left :=
  Budget.RowUpperBound.branch row11719_small_covers row11719_block_005_covers

def row11719_large_left_right_right : List Nat := row11719_block_007 ++ row11719_block_008
theorem row11719_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11719 row11719_large_left_right_right :=
  Budget.RowUpperBound.branch row11719_block_007_covers row11719_block_008_covers

def row11719_large_left_right : List Nat := row11719_block_006 ++ row11719_large_left_right_right
theorem row11719_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11719 row11719_large_left_right :=
  Budget.RowUpperBound.branch row11719_block_006_covers row11719_large_left_right_right_covers

def row11719_large_left : List Nat := row11719_large_left_left ++ row11719_large_left_right
theorem row11719_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11719 row11719_large_left :=
  Budget.RowUpperBound.branch row11719_large_left_left_covers row11719_large_left_right_covers

def row11719_large_right_left : List Nat := row11719_block_009 ++ row11719_block_010
theorem row11719_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11719 row11719_large_right_left :=
  Budget.RowUpperBound.branch row11719_block_009_covers row11719_block_010_covers

def row11719_large_right_right_right : List Nat := row11719_block_012 ++ row11719_block_013
theorem row11719_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11719 row11719_large_right_right_right :=
  Budget.RowUpperBound.branch row11719_block_012_covers row11719_block_013_covers

def row11719_large_right_right : List Nat := row11719_block_011 ++ row11719_large_right_right_right
theorem row11719_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11719 row11719_large_right_right :=
  Budget.RowUpperBound.branch row11719_block_011_covers row11719_large_right_right_right_covers

def row11719_large_right : List Nat := row11719_large_right_left ++ row11719_large_right_right
theorem row11719_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11719 row11719_large_right :=
  Budget.RowUpperBound.branch row11719_large_right_left_covers row11719_large_right_right_covers

def row11719_large : List Nat := row11719_large_left ++ row11719_large_right
theorem row11719_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11719 row11719_large :=
  Budget.RowUpperBound.branch row11719_large_left_covers row11719_large_right_covers

def row11719 : List Nat := row11719_large
theorem row11719_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11719 row11719 := row11719_large_covers

def row11731_small : List Nat := []
theorem row11731_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11731 row11731_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11731_block_005 : List Nat := []
theorem row11731_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11731 row11731_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11731_block_006 : List Nat := [11743]
def plan11731_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11731_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11731 row11731_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11731_block_006
  · decide
  · rfl

def row11731_block_007 : List Nat := [12613, 12637, 13147, 13729, 13999, 14083]
def plan11731_block_007 : List Budget.Choice := [(.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 19), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .second, .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 37), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .second, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, .possible, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11731_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11731 row11731_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11731_block_007
  · decide
  · rfl

def row11731_block_008 : List Nat := []
theorem row11731_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11731 row11731_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11731_block_009 : List Nat := []
theorem row11731_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11731 row11731_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11731_block_010 : List Nat := []
theorem row11731_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11731 row11731_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11731_block_011 : List Nat := []
theorem row11731_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11731 row11731_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11731_block_012 : List Nat := []
theorem row11731_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11731 row11731_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11731_block_013 : List Nat := []
theorem row11731_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11731 row11731_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11731_large_left_left : List Nat := row11731_small ++ row11731_block_005
theorem row11731_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11731 row11731_large_left_left :=
  Budget.RowUpperBound.branch row11731_small_covers row11731_block_005_covers

def row11731_large_left_right_right : List Nat := row11731_block_007 ++ row11731_block_008
theorem row11731_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11731 row11731_large_left_right_right :=
  Budget.RowUpperBound.branch row11731_block_007_covers row11731_block_008_covers

def row11731_large_left_right : List Nat := row11731_block_006 ++ row11731_large_left_right_right
theorem row11731_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11731 row11731_large_left_right :=
  Budget.RowUpperBound.branch row11731_block_006_covers row11731_large_left_right_right_covers

def row11731_large_left : List Nat := row11731_large_left_left ++ row11731_large_left_right
theorem row11731_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11731 row11731_large_left :=
  Budget.RowUpperBound.branch row11731_large_left_left_covers row11731_large_left_right_covers

def row11731_large_right_left : List Nat := row11731_block_009 ++ row11731_block_010
theorem row11731_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11731 row11731_large_right_left :=
  Budget.RowUpperBound.branch row11731_block_009_covers row11731_block_010_covers

def row11731_large_right_right_right : List Nat := row11731_block_012 ++ row11731_block_013
theorem row11731_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11731 row11731_large_right_right_right :=
  Budget.RowUpperBound.branch row11731_block_012_covers row11731_block_013_covers

def row11731_large_right_right : List Nat := row11731_block_011 ++ row11731_large_right_right_right
theorem row11731_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11731 row11731_large_right_right :=
  Budget.RowUpperBound.branch row11731_block_011_covers row11731_large_right_right_right_covers

def row11731_large_right : List Nat := row11731_large_right_left ++ row11731_large_right_right
theorem row11731_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11731 row11731_large_right :=
  Budget.RowUpperBound.branch row11731_large_right_left_covers row11731_large_right_right_covers

def row11731_large : List Nat := row11731_large_left ++ row11731_large_right
theorem row11731_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11731 row11731_large :=
  Budget.RowUpperBound.branch row11731_large_left_covers row11731_large_right_covers

def row11731 : List Nat := row11731_large
theorem row11731_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11731 row11731 := row11731_large_covers

def row11743_small : List Nat := []
theorem row11743_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11743 row11743_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11743_block_005 : List Nat := []
theorem row11743_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11743 row11743_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11743_block_006 : List Nat := [11953]
def plan11743_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .possible, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11743_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11743 row11743_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11743_block_006
  · decide
  · rfl

def row11743_block_007 : List Nat := [12487, 13009, 13177, 13669]
def plan11743_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 13), .second, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 31), (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .second, .possible, (.firstFactor 19), .first, .first, .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 37), (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11743_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11743 row11743_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11743_block_007
  · decide
  · rfl

def row11743_block_008 : List Nat := []
theorem row11743_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11743 row11743_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11743_block_009 : List Nat := []
theorem row11743_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11743 row11743_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11743_block_010 : List Nat := []
theorem row11743_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11743 row11743_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11743_block_011 : List Nat := []
theorem row11743_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11743 row11743_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11743_block_012 : List Nat := []
theorem row11743_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11743 row11743_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11743_block_013 : List Nat := []
theorem row11743_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11743 row11743_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11743_large_left_left : List Nat := row11743_small ++ row11743_block_005
theorem row11743_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11743 row11743_large_left_left :=
  Budget.RowUpperBound.branch row11743_small_covers row11743_block_005_covers

def row11743_large_left_right_right : List Nat := row11743_block_007 ++ row11743_block_008
theorem row11743_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11743 row11743_large_left_right_right :=
  Budget.RowUpperBound.branch row11743_block_007_covers row11743_block_008_covers

def row11743_large_left_right : List Nat := row11743_block_006 ++ row11743_large_left_right_right
theorem row11743_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11743 row11743_large_left_right :=
  Budget.RowUpperBound.branch row11743_block_006_covers row11743_large_left_right_right_covers

def row11743_large_left : List Nat := row11743_large_left_left ++ row11743_large_left_right
theorem row11743_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11743 row11743_large_left :=
  Budget.RowUpperBound.branch row11743_large_left_left_covers row11743_large_left_right_covers

def row11743_large_right_left : List Nat := row11743_block_009 ++ row11743_block_010
theorem row11743_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11743 row11743_large_right_left :=
  Budget.RowUpperBound.branch row11743_block_009_covers row11743_block_010_covers

def row11743_large_right_right_right : List Nat := row11743_block_012 ++ row11743_block_013
theorem row11743_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11743 row11743_large_right_right_right :=
  Budget.RowUpperBound.branch row11743_block_012_covers row11743_block_013_covers

def row11743_large_right_right : List Nat := row11743_block_011 ++ row11743_large_right_right_right
theorem row11743_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11743 row11743_large_right_right :=
  Budget.RowUpperBound.branch row11743_block_011_covers row11743_large_right_right_right_covers

def row11743_large_right : List Nat := row11743_large_right_left ++ row11743_large_right_right
theorem row11743_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11743 row11743_large_right :=
  Budget.RowUpperBound.branch row11743_large_right_left_covers row11743_large_right_right_covers

def row11743_large : List Nat := row11743_large_left ++ row11743_large_right
theorem row11743_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11743 row11743_large :=
  Budget.RowUpperBound.branch row11743_large_left_covers row11743_large_right_covers

def row11743 : List Nat := row11743_large
theorem row11743_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11743 row11743 := row11743_large_covers

def row11777_small : List Nat := []
theorem row11777_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11777 row11777_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11777_block_005 : List Nat := []
theorem row11777_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11777 row11777_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11777_block_006 : List Nat := [11867, 12119, 12203]
def plan11777_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .second, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), .first, .first, .second, .first, .first, .first, .possible, .first, (.secondFactor 23), (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .possible, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .possible, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row11777_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11777 row11777_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11777_block_006
  · decide
  · rfl

def row11777_block_007 : List Nat := [12917, 13331, 13679]
def plan11777_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .second, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 31), .possible, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 23), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 23), .first, .first, .second, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 17), .first, (.secondFactor 7), .second, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11777_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11777 row11777_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11777_block_007
  · decide
  · rfl

def row11777_block_008 : List Nat := []
theorem row11777_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11777 row11777_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11777_block_009 : List Nat := []
theorem row11777_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11777 row11777_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11777_block_010 : List Nat := []
theorem row11777_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11777 row11777_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11777_block_011 : List Nat := []
theorem row11777_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11777 row11777_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11777_block_012 : List Nat := []
theorem row11777_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11777 row11777_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11777_block_013 : List Nat := []
theorem row11777_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11777 row11777_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11777_large_left_left : List Nat := row11777_small ++ row11777_block_005
theorem row11777_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11777 row11777_large_left_left :=
  Budget.RowUpperBound.branch row11777_small_covers row11777_block_005_covers

def row11777_large_left_right_right : List Nat := row11777_block_007 ++ row11777_block_008
theorem row11777_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11777 row11777_large_left_right_right :=
  Budget.RowUpperBound.branch row11777_block_007_covers row11777_block_008_covers

def row11777_large_left_right : List Nat := row11777_block_006 ++ row11777_large_left_right_right
theorem row11777_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11777 row11777_large_left_right :=
  Budget.RowUpperBound.branch row11777_block_006_covers row11777_large_left_right_right_covers

def row11777_large_left : List Nat := row11777_large_left_left ++ row11777_large_left_right
theorem row11777_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11777 row11777_large_left :=
  Budget.RowUpperBound.branch row11777_large_left_left_covers row11777_large_left_right_covers

def row11777_large_right_left : List Nat := row11777_block_009 ++ row11777_block_010
theorem row11777_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11777 row11777_large_right_left :=
  Budget.RowUpperBound.branch row11777_block_009_covers row11777_block_010_covers

def row11777_large_right_right_right : List Nat := row11777_block_012 ++ row11777_block_013
theorem row11777_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11777 row11777_large_right_right_right :=
  Budget.RowUpperBound.branch row11777_block_012_covers row11777_block_013_covers

def row11777_large_right_right : List Nat := row11777_block_011 ++ row11777_large_right_right_right
theorem row11777_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11777 row11777_large_right_right :=
  Budget.RowUpperBound.branch row11777_block_011_covers row11777_large_right_right_right_covers

def row11777_large_right : List Nat := row11777_large_right_left ++ row11777_large_right_right
theorem row11777_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11777 row11777_large_right :=
  Budget.RowUpperBound.branch row11777_large_right_left_covers row11777_large_right_right_covers

def row11777_large : List Nat := row11777_large_left ++ row11777_large_right
theorem row11777_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11777 row11777_large :=
  Budget.RowUpperBound.branch row11777_large_left_covers row11777_large_right_covers

def row11777 : List Nat := row11777_large
theorem row11777_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11777 row11777 := row11777_large_covers

end ProjectEulerSolutions.PrimePairLarge
