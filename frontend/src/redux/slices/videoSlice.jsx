import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  video: null,
  banner: null,
}

const videoSlice = createSlice({
  name: 'video',
  initialState,
  reducers: {
    setVideo: (state, action) => {
      state.video = action.payload
    },
    setBanner: (state, action) => {
      state.banner = action.payload
    },
  },
})

export const { setVideo, setBanner } = videoSlice.actions

export default videoSlice.reducer