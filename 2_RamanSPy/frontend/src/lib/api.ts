import axios from 'axios';

// In production, use environment variable
const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

const api = axios.create({
    baseURL: BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const uploadFile = async (file: File, format: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_format', format);

    const response = await api.post('/data/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

export default api;
